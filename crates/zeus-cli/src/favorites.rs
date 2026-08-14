//! Favorite model persistence (`favorites.toml`, next to `keys.toml`).
//! Extracted from `tui.rs` so the small file-IO helpers live apart from
//! the TUI monolith.

use zeus_config::Config;

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct FavoritesFile {
    #[serde(default)]
    favorites: Vec<FavEntry>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct FavEntry {
    provider: String,
    model: String,
}

/// Lives next to `keys.toml` — same directory, same "small user-editable
/// TOML file in the zeus home dir" convention.
fn favorites_path(config: &Config) -> std::path::PathBuf {
    config
        .global
        .keys_toml
        .parent()
        .map(|p| p.join("favorites.toml"))
        .unwrap_or_else(|| std::path::PathBuf::from("favorites.toml"))
}

pub(crate) fn load_favorites(config: &Config) -> Vec<(String, String)> {
    let path = favorites_path(config);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    toml::from_str::<FavoritesFile>(&text)
        .map(|f| {
            f.favorites
                .into_iter()
                .map(|e| (e.provider, e.model))
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn save_favorites(config: &Config, favorites: &[(String, String)]) {
    let path = favorites_path(config);
    let file = FavoritesFile {
        favorites: favorites
            .iter()
            .map(|(provider, model)| FavEntry {
                provider: provider.clone(),
                model: model.clone(),
            })
            .collect(),
    };
    if let Ok(text) = toml::to_string_pretty(&file) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, text);
    }
}
