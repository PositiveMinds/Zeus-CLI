//! Model/provider picker rows, navigation, catalog grouping, and filtering -
//! the pure data side of `/model` and `/provider`. Extracted from `tui.rs`
//! so picker logic can be tested and reused without the TUI monolith.

use zeus_config::Config;
use zeus_provider::ModelInfo;

/// One row in the model picker: a non-selectable provider-group header, or
/// a selectable model belonging to that provider. Kept as a flat list (with
/// header rows navigation skips over) rather than nested groups, so a
/// single `ListState`/`selected` index still works for both keyboard and
/// mouse selection.
#[derive(Clone, Debug)]
pub(crate) enum PickerEntry {
    Header(String),
    /// A vendor-family grouping nested under a `Header` — e.g. "Anthropic"
    /// under the "OPENROUTER" provider header, when that provider's catalog
    /// spans more than one recognizable family. Non-selectable, same as
    /// `Header` — every "skip non-Model rows" navigation check already
    /// matches on `Model { .. }` specifically, so this needs no separate
    /// handling there.
    SubHeader(String),
    Model {
        provider: String,
        model: ModelInfo,
    },
}

/// One row in the provider picker: a non-selectable group header (paid /
/// free / local), a selectable provider, or a selectable model belonging to
/// that provider. Flat list so a single index drives keyboard and mouse
/// selection.
#[derive(Clone)]
pub(crate) enum ProviderEntry {
    /// A non-selectable category label ("Popular"/"Providers").
    Header(String),
    Provider {
        name: String,
        /// Short descriptor shown next to the name — see
        /// `provider_short_desc` (not the raw `providers.toml` `kind`).
        kind: String,
        /// True when the provider can be used right now (local kind, stored
        /// key, or env key present) — false means "needs a key" and Enter
        /// jumps to `KeyEntry` instead of switching. ctrl+k in the picker
        /// opens `KeyEntry` regardless, so an existing key can be updated.
        ready: bool,
    },
}

/// Moves `selected` one step in the given direction (`1` or `-1`), skipping
/// over header rows, wrapping around the ends. Safe as long as `entries`
/// contains at least one `Model` row (always true by construction — a
/// header is only ever pushed alongside its models).
pub(crate) fn picker_move(entries: &[PickerEntry], selected: usize, dir: isize) -> usize {
    let len = entries.len() as isize;
    if len == 0 {
        return 0;
    }
    let mut idx = selected as isize;
    loop {
        idx = (idx + dir).rem_euclid(len);
        if matches!(entries[idx as usize], PickerEntry::Model { .. }) {
            return idx as usize;
        }
    }
}

/// Same navigation for the provider picker, skipping its group headers but
/// allowing both provider and model rows to be selected.
pub(crate) fn provider_picker_move(
    entries: &[ProviderEntry],
    selected: usize,
    dir: isize,
) -> usize {
    let len = entries.len() as isize;
    if len == 0 {
        return 0;
    }
    let mut idx = selected as isize;
    loop {
        idx = (idx + dir).rem_euclid(len);
        if !matches!(entries[idx as usize], ProviderEntry::Header(_)) {
            return idx as usize;
        }
    }
}

/// Free-vs-paid tag for a fetched model id. Generous free heuristic — matches
/// the common free/low-cost tiers across providers (opencodezen's
/// deepseek-v4-flash-free, gemini flash, gpt mini, lite/nano variants, and
/// openrouter's `:free` suffixes) so as many genuinely free models as possible
/// surface as green in the picker.
pub(crate) fn is_free_model(id: &str) -> bool {
    let id = id.to_ascii_lowercase();
    const FREE_SUBSTRINGS: &[&str] = &[
        "free", "flash", "mini", "lite", "nano", "tiny", "light", "small", "1b", "3b", "8b",
    ];
    FREE_SUBSTRINGS.iter().any(|s| id.contains(s))
}

/// Infers the underlying model vendor/family from its id — used to further
/// Recognizes well-known public model-family prefixes, for sub-grouping an
/// aggregator provider's (OpenRouter, OpenCode Zen) catalog — which
/// otherwise dumps dozens of unrelated vendors' models under one flat list
/// with no way to tell a Claude model from a Gemini one at a glance.
/// Returns `None` for anything unrecognized rather than inventing a family
/// from the id itself: a provider's small in-house catalog (e.g. six
/// differently-named free models with no shared vendor) must stay one flat
/// list, not fragment into six one-model "families".
pub(crate) fn model_family(id: &str) -> Option<&'static str> {
    let lower = id.to_ascii_lowercase();
    // Aggregators commonly namespace ids as "vendor/model-name".
    let lower = lower.rsplit('/').next().unwrap_or(&lower);
    const FAMILIES: &[(&str, &str)] = &[
        ("claude", "Anthropic"),
        ("chatgpt", "OpenAI"),
        ("gpt", "OpenAI"),
        ("o1", "OpenAI"),
        ("o3", "OpenAI"),
        ("o4", "OpenAI"),
        ("gemini", "Google Gemini"),
        ("grok", "xAI"),
        ("deepseek", "DeepSeek"),
        ("glm", "Zhipu GLM"),
        ("minimax", "MiniMax"),
        ("llama", "Meta Llama"),
        ("qwen", "Qwen"),
        ("mixtral", "Mistral"),
        ("codestral", "Mistral"),
        ("mistral", "Mistral"),
        ("command", "Cohere"),
        ("phi", "Microsoft Phi"),
        ("sonar", "Perplexity"),
        ("nova", "Amazon Nova"),
        ("yi-", "01.AI"),
    ];
    FAMILIES
        .iter()
        .find(|(prefix, _)| lower.starts_with(prefix))
        .map(|(_, family)| *family)
}

/// Splits a provider's model catalog into vendor-family sub-groups for
/// display, sorted alphabetically by family name. Only kicks in once at
/// least two *recognized* families are actually present — a single-vendor
/// provider (Anthropic, OpenAI, …), or a small in-house catalog with no
/// recognizable big-lab prefixes at all, returns one `None`-keyed group
/// (meaning "no sub-header, render flat") instead. Models with no
/// recognized family are pooled into that same flat, header-less group —
/// they're never given their own singleton sub-header.
pub(crate) fn group_models_by_family(
    models: &[zeus_provider::ModelInfo],
) -> Vec<(Option<String>, Vec<zeus_provider::ModelInfo>)> {
    let recognized: std::collections::BTreeSet<&str> =
        models.iter().filter_map(|m| model_family(&m.id)).collect();
    if recognized.len() <= 1 {
        return vec![(None, models.to_vec())];
    }
    let mut unrecognized = Vec::new();
    let mut by_family: std::collections::BTreeMap<&str, Vec<zeus_provider::ModelInfo>> =
        std::collections::BTreeMap::new();
    for m in models {
        match model_family(&m.id) {
            Some(family) => by_family.entry(family).or_default().push(m.clone()),
            None => unrecognized.push(m.clone()),
        }
    }
    let mut groups: Vec<(Option<String>, Vec<zeus_provider::ModelInfo>)> = Vec::new();
    if !unrecognized.is_empty() {
        groups.push((None, unrecognized));
    }
    groups.extend(
        by_family
            .into_iter()
            .map(|(f, ms)| (Some(f.to_string()), ms)),
    );
    groups
}

/// Build grouped provider-picker entries: one header row per *provider*
/// (Anthropic, OpenRouter, OpenCode Zen, …) — the current provider leads,
/// the rest follow alphabetically, same ordering as the top-right dropdown.
/// Each provider's row carries its kind, default model, and whether it's
/// immediately usable (local kind, stored key, or env key set); its real
/// models (when reachable) are listed underneath, each individually tagged
/// free or paid — a single provider like OpenRouter can offer both. A
/// provider that can't list models (no key, server down) still shows as a
/// switchable row.
/// A few providers surfaced first under a "Popular" category, matching the
/// reference product's own priority list — everything else follows,
/// alphabetical, under a plain "Providers" category. Picking a provider no
/// longer shows its models inline (that's a separate, deliberate step —
/// see `apply_provider_picker_choice`/`persist_key_and_switch`'s auto-chain
/// into the model picker), so opening this list needs no network probe at
/// all and is instant regardless of how many providers are configured.
pub(crate) const POPULAR_PROVIDERS: [&str; 8] = [
    "opencodezen",
    "openrouter",
    "anthropic",
    "openai",
    "gemini",
    "groq",
    "cerebras",
    "deepseek",
];

/// A short, provider-specific descriptor shown next to its name — the
/// reference product bakes in strings like "(Recommended)"/"(API key)"
/// rather than leaving the row bare; unrecognized providers just fall back
/// to their `kind`.
pub(crate) fn provider_short_desc(name: &str, kind: &str) -> String {
    match name {
        "opencodezen" | "opencodezen-go" => "(Recommended)".to_string(),
        "openrouter" | "anthropic" | "openai" | "deepseek" | "mistral" | "groq" | "together"
        | "fireworks" | "charm" | "vercel" | "minimax" | "synthetic" | "huggingface" | "ionet"
        | "alibaba-sg" | "alibaba-us" | "avian" | "vertexai" | "bedrock" | "azure-openai"
        | "moonshot" | "zai" => "(API key)".to_string(),
        "ollama" | "lmstudio" | "llamacpp" => "(local)".to_string(),
        _ => format!("({kind})"),
    }
}

pub(crate) fn provider_picker_entries(
    config: &Config,
    current: &str,
) -> (Vec<ProviderEntry>, usize) {
    let mut names: Vec<&String> = config.providers.providers.keys().collect();
    names.sort();

    let mut popular: Vec<&String> = Vec::new();
    let mut rest: Vec<&String> = Vec::new();
    for name in names {
        if POPULAR_PROVIDERS.contains(&name.as_str()) {
            popular.push(name);
        } else {
            rest.push(name);
        }
    }
    popular.sort_by_key(|n| {
        POPULAR_PROVIDERS
            .iter()
            .position(|p| *p == n.as_str())
            .unwrap_or(usize::MAX)
    });

    let mut entries = Vec::new();
    let mut selected = 0;
    let mut push_group = |entries: &mut Vec<ProviderEntry>, title: &str, names: &[&String]| {
        if names.is_empty() {
            return;
        }
        entries.push(ProviderEntry::Header(title.to_string()));
        for name in names {
            let Some(cfg) = config.providers.get(name.as_str()) else {
                continue;
            };
            let ready = provider_status_ok(config, name);
            if name.as_str() == current {
                selected = entries.len();
            }
            entries.push(ProviderEntry::Provider {
                name: name.to_string(),
                kind: provider_short_desc(name, &cfg.kind),
                ready,
            });
        }
    };
    push_group(&mut entries, "Popular", &popular);
    push_group(&mut entries, "Providers", &rest);
    if selected == 0 {
        selected = provider_picker_move(&entries, 0, 1);
    }
    (entries, selected)
}

/// Backs `/model`'s no-argument picker: probes every configured provider,
/// leads with "Favorites" then "Recent" sections (each only shown when
/// non-empty, and only for models that actually came back from the scan),
/// puts the current provider's own group first among the rest (like
/// opencode's own picker), and flattens into `PickerEntry` rows with the
/// current model pre-selected. Empty result means "no models found".
pub(crate) fn build_model_picker_entries(
    models: &[(String, Vec<zeus_provider::ModelInfo>)],
    current_provider: &str,
    current_model: &str,
    recent: &[(String, String)],
    favorites: &[(String, String)],
) -> (Vec<PickerEntry>, usize) {
    let find = |provider: &str, model_id: &str| -> Option<zeus_provider::ModelInfo> {
        models
            .iter()
            .find(|(p, _)| p == provider)
            .and_then(|(_, ms)| ms.iter().find(|m| m.id == model_id))
            .cloned()
    };
    let mut entries = Vec::new();
    let mut selected = None;
    let push_section = |entries: &mut Vec<PickerEntry>,
                        selected: &mut Option<usize>,
                        title: &str,
                        pairs: &[(String, String)]| {
        let rows: Vec<(String, zeus_provider::ModelInfo)> = pairs
            .iter()
            .filter_map(|(p, m)| find(p, m).map(|mi| (p.clone(), mi)))
            .collect();
        if rows.is_empty() {
            return;
        }
        entries.push(PickerEntry::Header(title.to_string()));
        for (provider, model) in rows {
            if selected.is_none() && model.id == current_model && provider == current_provider {
                *selected = Some(entries.len());
            }
            entries.push(PickerEntry::Model { provider, model });
        }
    };
    push_section(&mut entries, &mut selected, "Favorites", favorites);
    push_section(&mut entries, &mut selected, "Recent", recent);

    let mut groups = models.to_vec();
    if let Some(pos) = groups.iter().position(|(name, _)| name == current_provider) {
        let current = groups.remove(pos);
        groups.insert(0, current);
    }
    for (provider_name, models) in groups {
        entries.push(PickerEntry::Header(provider_name.clone()));
        // Sub-group by vendor family for an aggregator (OpenRouter, OpenCode
        // Zen) whose catalog spans more than one recognizable vendor — see
        // `group_models_by_family`.
        for (family, family_models) in group_models_by_family(&models) {
            if let Some(family) = family {
                entries.push(PickerEntry::SubHeader(family));
            }
            for model in family_models {
                if selected.is_none()
                    && model.id == current_model
                    && provider_name == current_provider
                {
                    selected = Some(entries.len());
                }
                entries.push(PickerEntry::Model {
                    provider: provider_name.clone(),
                    model,
                });
            }
        }
    }
    let selected = selected.unwrap_or_else(|| first_selectable_picker(&entries));
    (entries, selected)
}

/// First `Model` row in a `PickerEntry` list — the default selection so
/// Enter works immediately without an arrow-key press first (index 0 is
/// always a `Header`).
pub(crate) fn first_selectable_picker(entries: &[PickerEntry]) -> usize {
    entries
        .iter()
        .position(|e| matches!(e, PickerEntry::Model { .. }))
        .unwrap_or(0)
}

/// Filter the `/model` picker's entries by `state.model_picker_search`
/// (matches on model name/id or provider name) — headers only survive with
/// an empty query, since a flat match list needs no group labels.
pub(crate) fn model_picker_filtered(entries: &[PickerEntry], search: &str) -> Vec<PickerEntry> {
    let q = search.to_lowercase();
    entries
        .iter()
        .filter(|e| match e {
            PickerEntry::Header(_) | PickerEntry::SubHeader(_) => q.is_empty(),
            PickerEntry::Model { provider, model } => {
                q.is_empty()
                    || model.name.to_lowercase().contains(&q)
                    || model.id.to_lowercase().contains(&q)
                    || provider.to_lowercase().contains(&q)
            }
        })
        .cloned()
        .collect()
}

/// The provider dot in the top-right button and pickers: green when the
pub(crate) fn provider_status_ok(config: &Config, provider: &str) -> bool {
    let Some(cfg) = config.providers.get(provider) else {
        return false;
    };
    if matches!(cfg.kind.as_str(), "ollama" | "lmstudio" | "llamacpp")
        || cfg.headers.contains_key("Authorization")
    {
        return true;
    }
    match &cfg.api_key_env {
        Some(var) => std::env::var(var).map(|k| !k.is_empty()).unwrap_or(false),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mi(id: &str) -> zeus_provider::ModelInfo {
        zeus_provider::ModelInfo {
            id: id.to_string(),
            name: id.to_string(),
            context_window: None,
        }
    }

    #[test]
    fn model_family_maps_known_vendors() {
        assert_eq!(model_family("gpt-4o"), Some("OpenAI"));
        assert_eq!(model_family("openai/gpt-4o-mini"), Some("OpenAI"));
        assert_eq!(model_family("claude-3-5-sonnet"), Some("Anthropic"));
        assert_eq!(model_family("gemini-2.0-flash"), Some("Google Gemini"));
        assert_eq!(model_family("deepseek-chat"), Some("DeepSeek"));
        assert_eq!(model_family("a-custom-model"), None);
    }

    #[test]
    fn multi_vendor_catalog_groups_by_family() {
        let models = vec![mi("gpt-4o"), mi("claude-3-5-sonnet")];
        let groups = group_models_by_family(&models);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].0.as_deref(), Some("Anthropic"));
        assert_eq!(groups[1].0.as_deref(), Some("OpenAI"));
    }

    #[test]
    fn single_vendor_catalog_stays_flat() {
        let models = vec![mi("gpt-4o"), mi("gpt-4o-mini")];
        let groups = group_models_by_family(&models);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].0.is_none());
        assert_eq!(groups[0].1.len(), 2);
    }

    #[test]
    fn model_picker_puts_favorites_first_and_selects_current() {
        let models: Vec<(String, Vec<zeus_provider::ModelInfo>)> =
            vec![("openai".to_string(), vec![mi("gpt-4o"), mi("gpt-4o-mini")])];
        let recent = vec![("openai".to_string(), "gpt-4o-mini".to_string())];
        let favorites = vec![("openai".to_string(), "gpt-4o".to_string())];
        let (entries, selected) =
            build_model_picker_entries(&models, "openai", "gpt-4o", &recent, &favorites);
        // Favorites section header, then the favorited model, then Recent…
        assert!(matches!(entries[0], PickerEntry::Header(_)));
        match &entries[1] {
            PickerEntry::Model { model, .. } => assert_eq!(model.id, "gpt-4o"),
            other => panic!("expected Model row, got {other:?}"),
        }
        // The current model (gpt-4o) is selected in the favorites section.
        assert!(
            matches!(&entries[selected], PickerEntry::Model { model, .. } if model.id == "gpt-4o")
        );
    }
}
