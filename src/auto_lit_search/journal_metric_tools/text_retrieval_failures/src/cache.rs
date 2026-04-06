use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Clone, Default, Serialize)]
pub struct CacheEntry {
    pub publisher: String,
    pub journal: String,
}

pub fn load_cache(path: &Path) -> HashMap<String, CacheEntry> {
    let Ok(f) = File::open(path) else {
        return HashMap::new();
    };
    let Ok(val): Result<Value, _> = serde_json::from_reader(f) else {
        return HashMap::new();
    };
    let Some(obj) = val.as_object() else {
        return HashMap::new();
    };
    let mut out = HashMap::new();
    for (k, v) in obj {
        let pub_s = v
            .get("publisher")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let jour = v
            .get("journal")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        out.insert(
            k.clone(),
            CacheEntry {
                publisher: pub_s,
                journal: jour,
            },
        );
    }
    out
}

pub fn save_cache(path: &Path, cache: &HashMap<String, CacheEntry>) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut map: HashMap<&str, &CacheEntry> = HashMap::new();
    for (k, v) in cache {
        map.insert(k.as_str(), v);
    }
    let f = File::create(path).with_context(|| format!("write {}", path.display()))?;
    serde_json::to_writer_pretty(f, &map)?;
    Ok(())
}
