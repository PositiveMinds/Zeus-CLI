//! Phase 6 — tree-sitter AST extraction for the database-free symbol index.
//!
//! The regex extractors in [`crate::codeint`] are the baseline; this module
//! layers real tree-sitter parsing on top for the languages that have a wired
//! grammar. Each language declares:
//!
//! - which file extensions it owns, and
//! - a [`TsLang::map`] function that turns a parsed AST node into an optional
//!   `(name, kind)` pair, mirroring the kind vocabulary of the on-disk
//!   `Symbol` contract (function/class/struct/enum/interface/trait/type/const/
//!   var/method/module/static).
//!
//! `extract_symbols_ts` walks every named node once with a `TreeCursor` (no
//! recursion, so deeply nested files can't overflow the stack) and emits a
//! `Symbol` at `start_position().row + 1` for each mapping hit. Languages
//! without a grammar here keep the regex fallback unchanged, so the index
//! always has a baseline.

use std::path::Path;

use tree_sitter::{Language, Node, Parser};
use tree_sitter_language::LanguageFn;

use crate::codeint::Symbol;

/// Which field carries the identifier for a symbol node.
fn field_name<'t>(node: &Node<'t>) -> Option<Node<'t>> {
    node.child_by_field_name("name")
}

/// Text of a node, when it's valid UTF-8.
fn text(node: &Node, src: &[u8]) -> Option<String> {
    node.utf8_text(src).ok().map(|s| s.to_string())
}

/// Generic `(node_kind, symbol_kind)` mapping with a `name` field lookup.
fn rule(node: &Node, src: &[u8], table: &[(&str, &str)]) -> Option<(String, String)> {
    let (_, kind) = table.iter().find(|(k, _)| *k == node.kind())?;
    let name = text(&field_name(node)?, src)?;
    Some((name, kind.to_string()))
}

/// Descend `declarator`/`name` fields to the identifier of a C-style
/// declarator (handles `int main`, `int *fp`, `typedef ... Foo` chains).
fn declarator_identifier<'t>(node: &Node<'t>, src: &[u8]) -> Option<String> {
    let mut cur = *node;
    for _ in 0..8 {
        if let Some(c) = cur.child_by_field_name("name") {
            if let Some(t) = text(&c, src) {
                return Some(t);
            }
        }
        match cur.kind() {
            "identifier" | "type_identifier" | "field_identifier" => {
                return text(&cur, src);
            }
            _ => {
                cur = cur.child_by_field_name("declarator")?;
            }
        }
    }
    None
}

fn map_rust(node: &Node, src: &[u8]) -> Option<(String, String)> {
    const T: &[(&str, &str)] = &[
        ("function_item", "function"),
        ("function_signature_item", "function"),
        ("struct_item", "struct"),
        ("enum_item", "enum"),
        ("trait_item", "trait"),
        ("type_item", "type"),
        ("const_item", "const"),
        ("static_item", "static"),
        ("mod_item", "module"),
    ];
    rule(node, src, T)
}

fn map_python(node: &Node, src: &[u8]) -> Option<(String, String)> {
    const T: &[(&str, &str)] = &[
        ("function_definition", "function"),
        ("class_definition", "class"),
    ];
    rule(node, src, T)
}

fn map_go(node: &Node, src: &[u8]) -> Option<(String, String)> {
    match node.kind() {
        "function_declaration" => Some((text(&field_name(node)?, src)?, "function".into())),
        "method_declaration" => Some((text(&field_name(node)?, src)?, "method".into())),
        "type_spec" => {
            let n = text(&field_name(node)?, src)?;
            let kind = match node.child_by_field_name("type") {
                Some(t) if t.kind() == "struct_type" => "struct",
                Some(t) if t.kind() == "interface_type" => "interface",
                _ => "type",
            };
            Some((n, kind.into()))
        }
        "const_spec" => Some((text(&field_name(node)?, src)?, "const".into())),
        "var_spec" => Some((text(&field_name(node)?, src)?, "var".into())),
        _ => None,
    }
}

fn map_js(node: &Node, src: &[u8]) -> Option<(String, String)> {
    match node.kind() {
        "function_declaration" | "generator_function_declaration" => {
            Some((text(&field_name(node)?, src)?, "function".into()))
        }
        "class_declaration" => Some((text(&field_name(node)?, src)?, "class".into())),
        "method_definition" => Some((text(&field_name(node)?, src)?, "method".into())),
        "interface_declaration" => Some((text(&field_name(node)?, src)?, "interface".into())),
        "enum_declaration" => Some((text(&field_name(node)?, src)?, "enum".into())),
        "type_alias_declaration" => Some((text(&field_name(node)?, src)?, "type".into())),
        "variable_declarator" => {
            let n = text(&field_name(node)?, src)?;
            let is_fn = matches!(
                node.child_by_field_name("value").map(|v| v.kind()),
                Some("arrow_function" | "function_expression")
            );
            Some((n, if is_fn { "function" } else { "const" }.to_string()))
        }
        _ => None,
    }
}

fn map_c(node: &Node, src: &[u8]) -> Option<(String, String)> {
    match node.kind() {
        "struct_specifier" => Some((text(&field_name(node)?, src)?, "struct".into())),
        "union_specifier" => Some((text(&field_name(node)?, src)?, "struct".into())),
        "enum_specifier" => Some((text(&field_name(node)?, src)?, "enum".into())),
        "type_definition" => Some((declarator_identifier(node, src)?, "type".into())),
        "function_declarator" => {
            let id = node.child_by_field_name("declarator")?;
            if id.kind() == "identifier" {
                Some((text(&id, src)?, "function".into()))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn map_cpp(node: &Node, src: &[u8]) -> Option<(String, String)> {
    match node.kind() {
        "class_specifier" => Some((text(&field_name(node)?, src)?, "class".into())),
        "struct_specifier" => Some((text(&field_name(node)?, src)?, "struct".into())),
        "union_specifier" => Some((text(&field_name(node)?, src)?, "struct".into())),
        "enum_specifier" => Some((text(&field_name(node)?, src)?, "enum".into())),
        "namespace_definition" => Some((text(&field_name(node)?, src)?, "module".into())),
        "alias_declaration" => Some((text(&field_name(node)?, src)?, "type".into())),
        "concept_definition" => Some((text(&field_name(node)?, src)?, "trait".into())),
        "type_definition" => Some((declarator_identifier(node, src)?, "type".into())),
        "function_declarator" => {
            let id = node.child_by_field_name("declarator")?;
            if matches!(
                id.kind(),
                "identifier" | "field_identifier" | "qualified_identifier"
            ) {
                Some((text(&id, src)?, "function".into()))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn map_java(node: &Node, src: &[u8]) -> Option<(String, String)> {
    const T: &[(&str, &str)] = &[
        ("class_declaration", "class"),
        ("interface_declaration", "interface"),
        ("enum_declaration", "enum"),
        ("record_declaration", "class"),
        ("method_declaration", "method"),
        ("constructor_declaration", "method"),
    ];
    rule(node, src, T)
}

fn map_csharp(node: &Node, src: &[u8]) -> Option<(String, String)> {
    const T: &[(&str, &str)] = &[
        ("class_declaration", "class"),
        ("interface_declaration", "interface"),
        ("struct_declaration", "struct"),
        ("enum_declaration", "enum"),
        ("record_declaration", "class"),
        ("method_declaration", "method"),
        ("namespace_declaration", "module"),
        ("file_scoped_namespace_declaration", "module"),
    ];
    rule(node, src, T)
}

fn map_ruby(node: &Node, src: &[u8]) -> Option<(String, String)> {
    match node.kind() {
        "class" => Some((text(&field_name(node)?, src)?, "class".into())),
        "module" => Some((text(&field_name(node)?, src)?, "module".into())),
        "method" | "singleton_method" | "setter" => {
            Some((text(&field_name(node)?, src)?, "method".into()))
        }
        _ => None,
    }
}

/// A parsed AST node mapped to a `(name, kind)` symbol.
type MappedSymbol = fn(&Node, &[u8]) -> Option<(String, String)>;

/// One language's tree-sitter wiring.
pub struct TsLang {
    /// File extensions this grammar owns (lowercase, no dot).
    pub exts: &'static [&'static str],
    /// The compiled grammar (stored as the zero-cost `LanguageFn` shim).
    language: LanguageFn,
    /// Node → `(name, kind)` extraction.
    map: MappedSymbol,
}

/// Languages with a wired tree-sitter grammar.
pub static TS_LANGS: &[TsLang] = &[
    TsLang {
        exts: &["rs"],
        language: tree_sitter_rust::LANGUAGE,
        map: map_rust,
    },
    TsLang {
        exts: &["py"],
        language: tree_sitter_python::LANGUAGE,
        map: map_python,
    },
    TsLang {
        exts: &["go"],
        language: tree_sitter_go::LANGUAGE,
        map: map_go,
    },
    TsLang {
        exts: &["js", "jsx", "mjs", "cjs"],
        language: tree_sitter_javascript::LANGUAGE,
        map: map_js,
    },
    TsLang {
        exts: &["ts"],
        language: tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
        map: map_js,
    },
    TsLang {
        exts: &["tsx"],
        language: tree_sitter_typescript::LANGUAGE_TSX,
        map: map_js,
    },
    TsLang {
        exts: &["c", "h"],
        language: tree_sitter_c::LANGUAGE,
        map: map_c,
    },
    TsLang {
        exts: &["cpp", "cc", "cxx", "hpp", "hh", "hxx"],
        language: tree_sitter_cpp::LANGUAGE,
        map: map_cpp,
    },
    TsLang {
        exts: &["java"],
        language: tree_sitter_java::LANGUAGE,
        map: map_java,
    },
    TsLang {
        exts: &["cs"],
        language: tree_sitter_c_sharp::LANGUAGE,
        map: map_csharp,
    },
    TsLang {
        exts: &["rb"],
        language: tree_sitter_ruby::LANGUAGE,
        map: map_ruby,
    },
];

/// Pick the tree-sitter language owning `path`, if any.
pub fn ts_language_for(path: &Path) -> Option<&'static TsLang> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_lowercase();
    TS_LANGS.iter().find(|l| l.exts.iter().any(|e| *e == ext))
}

/// Parse `text` with `lang` and append every mapped symbol to `out`.
pub fn extract_symbols_ts(rel: &str, lang: &TsLang, text: &str, out: &mut Vec<Symbol>) {
    let mut parser = Parser::new();
    let language: Language = lang.language.into();
    if parser.set_language(&language).is_err() {
        return;
    }
    let Some(tree) = parser.parse(text, None) else {
        return;
    };
    let src = text.as_bytes();
    let mut cursor = tree.walk();
    loop {
        let node = cursor.node();
        if let Some((name, kind)) = (lang.map)(&node, src) {
            out.push(Symbol {
                name,
                kind,
                file: rel.to_string(),
                line: node.start_position().row + 1,
            });
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(text: &str, path: &str) -> Vec<Symbol> {
        let mut out = Vec::new();
        let lang = ts_language_for(Path::new(path)).expect("grammar wired");
        extract_symbols_ts(&path.replace('\\', "/"), lang, text, &mut out);
        out
    }

    fn names(out: &[Symbol]) -> Vec<&str> {
        out.iter().map(|s| s.name.as_str()).collect()
    }

    #[test]
    fn rust_functions_structs_and_impls() {
        let text = "fn main() {}\nstruct Thing {}\nimpl Thing {\n  fn helper(&self) {}\n}\nenum E {}\ntrait T {}\n";
        let out = extract(text, "lib.rs");
        assert!(names(&out).contains(&"main"));
        assert!(names(&out).contains(&"Thing"));
        assert!(names(&out).contains(&"helper"));
        assert!(names(&out).contains(&"E"));
        assert!(names(&out).contains(&"T"));
        assert!(out.iter().all(|s| s.line >= 1));
        let main = out.iter().find(|s| s.name == "main").unwrap();
        assert_eq!(main.kind, "function");
        assert_eq!(main.line, 1);
    }

    #[test]
    fn python_class_and_methods() {
        let out = extract(
            "class Bot:\n    def talk(self):\n        return 1\n\nasync def main():\n    pass\n",
            "bot.py",
        );
        assert!(names(&out).contains(&"Bot"));
        assert!(names(&out).contains(&"talk"));
        assert!(names(&out).contains(&"main"));
        let bot = out.iter().find(|s| s.name == "Bot").unwrap();
        assert_eq!(bot.kind, "class");
    }

    #[test]
    fn go_type_spec_distinguishes_struct_interface() {
        let out = extract(
            "package p\n\ntype Point struct {\n  X int\n}\n\ntype Shape interface {\n  Area() float64\n}\n\nfunc New() *Point {\n  return &Point{}\n}\n\nvar version = \"1\"\n",
            "p.go",
        );
        let point = out.iter().find(|s| s.name == "Point").unwrap();
        assert_eq!(point.kind, "struct");
        let shape = out.iter().find(|s| s.name == "Shape").unwrap();
        assert_eq!(shape.kind, "interface");
        let new_fn = out.iter().find(|s| s.name == "New").unwrap();
        assert_eq!(new_fn.kind, "function");
        assert!(names(&out).contains(&"version"));
    }

    #[test]
    fn typescript_arrow_fn_const_class_interface() {
        let out = extract(
            "export interface User { id: number }\nconst greet = (n: string) => n;\nconst name = 'zeus';\nexport class Runner {}\nenum Level { Low }\nfunction helper() {}\n",
            "a.ts",
        );
        let greet = out.iter().find(|s| s.name == "greet").unwrap();
        assert_eq!(greet.kind, "function");
        let name = out.iter().find(|s| s.name == "name").unwrap();
        assert_eq!(name.kind, "const");
        assert!(out
            .iter()
            .any(|s| s.name == "User" && s.kind == "interface"));
        assert!(out.iter().any(|s| s.name == "Runner" && s.kind == "class"));
        assert!(out.iter().any(|s| s.name == "Level" && s.kind == "enum"));
        assert!(out
            .iter()
            .any(|s| s.name == "helper" && s.kind == "function"));
    }

    #[test]
    fn c_functions_and_types() {
        let out = extract(
            "#include <stdio.h>\nstruct Point { int x; };\nint add(int a, int b) {\n  return a + b;\n}\ntypedef unsigned int uint;\n",
            "util.c",
        );
        assert!(names(&out).contains(&"Point"));
        assert!(names(&out).contains(&"add"));
        assert!(names(&out).contains(&"uint"));
        let add = out.iter().find(|s| s.name == "add").unwrap();
        assert_eq!(add.kind, "function");
    }

    #[test]
    fn cpp_classes_namespaces_and_aliases() {
        let out = extract(
            "namespace app {\nclass Widget {\npublic:\n  void render();\n};\n}\nusing Alias = Widget;\n",
            "w.cpp",
        );
        assert!(names(&out).contains(&"app"));
        assert!(names(&out).contains(&"Widget"));
        assert!(names(&out).contains(&"Alias"));
        let widget = out.iter().find(|s| s.name == "Widget").unwrap();
        assert_eq!(widget.kind, "class");
    }

    #[test]
    fn java_and_csharp_types_and_methods() {
        let java = extract(
            "public class App {\n  public static void main(String[] args) {}\n}\ninterface Repo {}\n",
            "App.java",
        );
        assert!(names(&java).contains(&"App"));
        assert!(names(&java).contains(&"main"));
        assert!(java
            .iter()
            .any(|s| s.name == "Repo" && s.kind == "interface"));

        let cs = extract(
            "namespace Z;\npublic class Service {\n  public int Run() => 1;\n}\n",
            "S.cs",
        );
        assert!(names(&cs).contains(&"Z"));
        assert!(names(&cs).contains(&"Service"));
        assert!(names(&cs).contains(&"Run"));
    }

    #[test]
    fn ruby_classes_methods_and_singletons() {
        let out = extract(
            "module Store\nclass Cart\n  def total; 1; end\n  def self.of; end\nend\nend\n",
            "cart.rb",
        );
        assert!(names(&out).contains(&"Store"));
        assert!(names(&out).contains(&"Cart"));
        assert!(names(&out).contains(&"total"));
        assert!(out.iter().any(|s| s.name == "of" && s.kind == "method"));
    }

    #[test]
    fn malformed_input_degrades_gracefully() {
        let out = extract("fn broken( {\nstruct {\n", "lib.rs");
        assert!(out.iter().all(|s| s.line >= 1));
    }

    #[test]
    fn unsupported_extension_has_no_grammar() {
        assert!(ts_language_for(Path::new("x.swift")).is_none());
        assert!(ts_language_for(Path::new("x.lua")).is_none());
        assert!(ts_language_for(Path::new("x.kt")).is_none());
    }
}
