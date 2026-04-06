use serde_json::Value;
use std::fs;
use std::path::Path;

pub fn safe_stem_paper_id(paper_id: &str) -> String {
    paper_id
        .trim()
        .replace('/', "_")
        .replace(':', "_")
        .replace(' ', "_")
}

fn has_txt_prefix(dir: &Path, stem: &str) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    for e in entries.flatten() {
        let p = e.path();
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !name.ends_with(".txt") {
            continue;
        }
        if name.starts_with(stem) {
            if let Ok(md) = e.metadata() {
                if md.len() > 0 {
                    return true;
                }
            }
        }
    }
    false
}

fn n_successful_sources(rec: &Value) -> Option<u64> {
    rec.get("n_successful_sources")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            rec.get("details")
                .and_then(|d| d.get("n_successful_sources"))
                .and_then(|v| v.as_u64())
        })
}

pub fn has_text(rec: &Value, require_file: bool, manifest_path: &Path) -> bool {
    if let Some(n) = n_successful_sources(rec) {
        return n > 0;
    }
    if require_file {
        let adir = manifest_path.parent().unwrap_or(Path::new("."));
        let paper_id = rec.get("paper_id").and_then(|v| v.as_str()).unwrap_or("");
        let stem = safe_stem_paper_id(paper_id);
        if stem.is_empty() {
            return false;
        }
        has_txt_prefix(adir, &stem)
    } else {
        true
    }
}
