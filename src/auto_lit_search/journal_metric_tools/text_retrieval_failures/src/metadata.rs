use crate::cache::CacheEntry;
use crate::doi_prefix_hints;
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use tokio::sync::Mutex;

const EUROPEPMC_SEARCH: &str = "https://www.ebi.ac.uk/europepmc/webservices/rest/search";
const CROSSREF_PREFIX: &str = "https://api.crossref.org/works/";
pub const DEFAULT_UA: &str = "auto-lit-metrics/1.0 (text_retrieval_failures; mailto:none)";

#[derive(Clone, Debug)]
pub enum IdKind {
    Doi(String),
    Pmid(String),
    Pmcid(String),
    Epmc(String),
    Unknown(#[allow(dead_code)] String),
}

pub fn parse_id_for_lookup(paper_id: &str, details: &Value) -> IdKind {
    let raw = paper_id.trim();
    if let Some(d) = details.get("doi").and_then(|v| v.as_str()) {
        let d = d.trim();
        if !d.is_empty() {
            return IdKind::Doi(d.to_lowercase());
        }
    }
    if raw.to_uppercase().starts_with("DOI:") {
        let d = raw[4..].trim();
        if !d.is_empty() {
            return IdKind::Doi(d.to_lowercase());
        }
    }
    if raw.starts_with("10.") {
        return IdKind::Doi(raw.to_lowercase());
    }
    let re_pmid = Regex::new(r"(?i)^PMID:\s*(\d+)\s*$").unwrap();
    if let Some(c) = re_pmid.captures(raw) {
        return IdKind::Pmid(c.get(1).unwrap().as_str().to_string());
    }
    let re_pmc = Regex::new(r"(?i)^(PMC\d+)$").unwrap();
    if let Some(c) = re_pmc.captures(raw) {
        return IdKind::Pmcid(c.get(1).unwrap().as_str().to_uppercase());
    }
    let re_epmc = Regex::new(r"(?i)^EPMC:(\S+)$").unwrap();
    if let Some(c) = re_epmc.captures(raw) {
        return IdKind::Epmc(c.get(1).unwrap().as_str().to_string());
    }
    let re_digits = Regex::new(r"^\d+$").unwrap();
    if re_digits.is_match(raw) {
        return IdKind::Pmid(raw.to_string());
    }
    IdKind::Unknown(raw.to_string())
}

fn crossref_url(doi: &str) -> String {
    format!("{}{}", CROSSREF_PREFIX, doi.replace('/', "%2F"))
}

pub async fn crossref_lookup(
    client: &reqwest::Client,
    doi: &str,
    cache: &Mutex<HashMap<String, CacheEntry>>,
) -> (String, String) {
    let key = format!("crossref:{}", doi);
    {
        let g = cache.lock().await;
        if let Some(e) = g.get(&key) {
            return (e.publisher.clone(), e.journal.clone());
        }
    }
    let url = crossref_url(doi);
    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(_) => {
            let mut g = cache.lock().await;
            g.insert(
                key,
                CacheEntry {
                    publisher: String::new(),
                    journal: String::new(),
                },
            );
            return (String::new(), String::new());
        }
    };
    if !resp.status().is_success() {
        let mut g = cache.lock().await;
        g.insert(
            key,
            CacheEntry {
                publisher: String::new(),
                journal: String::new(),
            },
        );
        return (String::new(), String::new());
    }
    let Ok(val) = resp.json::<Value>().await else {
        let mut g = cache.lock().await;
        g.insert(
            key,
            CacheEntry {
                publisher: String::new(),
                journal: String::new(),
            },
        );
        return (String::new(), String::new());
    };
    let msg = val.get("message").cloned().unwrap_or(Value::Null);
    let pub_s = msg
        .get("publisher")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let jour = match msg.get("container-title") {
        Some(Value::Array(a)) => a.get(0).and_then(|v| v.as_str()).unwrap_or(""),
        Some(Value::String(s)) => s.as_str(),
        _ => "",
    }
    .to_string();
    let ent = CacheEntry {
        publisher: pub_s.clone(),
        journal: jour.clone(),
    };
    let mut g = cache.lock().await;
    g.insert(key, ent);
    (pub_s, jour)
}

fn epmc_query(kind: &str, value: &str) -> Option<String> {
    Some(match kind {
        "doi" => format!("DOI:\"{}\"", value),
        "pmid" => format!("SRC:MED AND EXT_ID:{}", value),
        "pmcid" => value.to_string(),
        "epmc" => format!("EXT_ID:{}", value),
        _ => return None,
    })
}

pub async fn epmc_lookup(
    client: &reqwest::Client,
    kind: &str,
    value: &str,
    cache: &Mutex<HashMap<String, CacheEntry>>,
) -> (String, String) {
    let key = format!("epmc:{}:{}", kind, value);
    {
        let g = cache.lock().await;
        if let Some(e) = g.get(&key) {
            return (e.publisher.clone(), e.journal.clone());
        }
    }
    let Some(query) = epmc_query(kind, value) else {
        return (String::new(), String::new());
    };
    let resp = match client
        .get(EUROPEPMC_SEARCH)
        .query(&[
            ("query", query.as_str()),
            ("format", "json"),
            ("resultType", "core"),
            ("pageSize", "5"),
        ])
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => {
            let mut g = cache.lock().await;
            g.insert(
                key,
                CacheEntry {
                    publisher: String::new(),
                    journal: String::new(),
                },
            );
            return (String::new(), String::new());
        }
    };
    if !resp.status().is_success() {
        let mut g = cache.lock().await;
        g.insert(
            key,
            CacheEntry {
                publisher: String::new(),
                journal: String::new(),
            },
        );
        return (String::new(), String::new());
    }
    let Ok(data) = resp.json::<Value>().await else {
        let mut g = cache.lock().await;
        g.insert(
            key,
            CacheEntry {
                publisher: String::new(),
                journal: String::new(),
            },
        );
        return (String::new(), String::new());
    };
    let n = data
        .get("hitCount")
        .and_then(|v| v.as_u64().or_else(|| v.as_i64().map(|i| i as u64)))
        .unwrap_or(0);
    if n == 0 {
        let mut g = cache.lock().await;
        g.insert(
            key,
            CacheEntry {
                publisher: String::new(),
                journal: String::new(),
            },
        );
        return (String::new(), String::new());
    }
    let rec = data
        .pointer("/resultList/result/0")
        .cloned()
        .unwrap_or(Value::Null);
    let ji = rec.get("journalInfo").cloned().unwrap_or(Value::Null);
    let title = ji
        .get("journal")
        .and_then(|j| j.get("title"))
        .and_then(|v| v.as_str())
        .or_else(|| ji.get("journalTitle").and_then(|v| v.as_str()))
        .unwrap_or("");
    let pub_s = ji.get("publisher").and_then(|v| v.as_str()).unwrap_or("");
    let ent = CacheEntry {
        publisher: pub_s.to_string(),
        journal: title.to_string(),
    };
    let out = (ent.publisher.clone(), ent.journal.clone());
    let mut g = cache.lock().await;
    g.insert(key, ent);
    out
}

pub async fn resolve_labels(
    paper_id: &str,
    details: &Value,
    client: &reqwest::Client,
    cache: &Mutex<HashMap<String, CacheEntry>>,
) -> (String, String) {
    match parse_id_for_lookup(paper_id, details) {
        IdKind::Doi(doi) => {
            let (mut pub_s, mut jour) = crossref_lookup(client, &doi, cache).await;
            if pub_s.is_empty() || jour.is_empty() {
                let (p2, j2) = epmc_lookup(client, "doi", &doi, cache).await;
                if pub_s.is_empty() {
                    pub_s = p2;
                }
                if jour.is_empty() {
                    jour = j2;
                }
            }
            if pub_s.is_empty() {
                if let Some(h) = doi_prefix_hints::publisher_for_doi(&doi) {
                    pub_s = h.to_string();
                }
            }
            (pub_s, jour)
        }
        IdKind::Pmid(v) => epmc_lookup(client, "pmid", &v, cache).await,
        IdKind::Pmcid(v) => epmc_lookup(client, "pmcid", &v, cache).await,
        IdKind::Epmc(v) => epmc_lookup(client, "epmc", &v, cache).await,
        IdKind::Unknown(_) => (String::new(), String::new()),
    }
}
