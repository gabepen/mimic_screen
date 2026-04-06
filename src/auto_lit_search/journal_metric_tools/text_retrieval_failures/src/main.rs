mod cache;
mod cli;
mod doi_prefix_hints;
mod metadata;
mod scan;

use anyhow::{Context, Result};
use cache::{load_cache, save_cache, CacheEntry};
use clap::Parser;
use cli::Args;
use futures::stream::{self, StreamExt};
use metadata::{resolve_labels, DEFAULT_UA};
use scan::has_text;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
struct FailureRow {
    paper_id: String,
    status: String,
    message: String,
    details: Value,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let jsonl_files = cli::collect_jsonl_paths(&args)?;
    if jsonl_files.is_empty() {
        anyhow::bail!("No .jsonl files found");
    }

    let mut total: u64 = 0;
    let mut failures: Vec<FailureRow> = Vec::new();

    for jf in &jsonl_files {
        let f = File::open(jf).with_context(|| format!("open {}", jf.display()))?;
        for line in BufReader::new(f).lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(rec) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            total += 1;
            if has_text(&rec, args.require_existing_file, jf) {
                continue;
            }
            let paper_id = rec
                .get("paper_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let status = rec
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let message = rec
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let details = rec.get("details").cloned().unwrap_or(Value::Null);
            failures.push(FailureRow {
                paper_id,
                status,
                message,
                details,
            });
        }
    }

    let disk_cache: HashMap<String, CacheEntry> = args
        .cache
        .as_ref()
        .map(|p| load_cache(p))
        .unwrap_or_default();
    let cache: Arc<Mutex<HashMap<String, CacheEntry>>> = Arc::new(Mutex::new(disk_cache));

    let client = reqwest::Client::builder()
        .user_agent(DEFAULT_UA)
        .build()?;

    let n_failed = failures.len();
    let conc = args.concurrency.max(1) as usize;

    let resolved: Vec<(String, String, String, String, String)> = if args.no_resolve {
        failures
            .iter()
            .map(|r| {
                (
                    r.paper_id.clone(),
                    r.status.clone(),
                    r.message.clone(),
                    String::new(),
                    String::new(),
                )
            })
            .collect()
    } else {
        stream::iter(failures.into_iter())
            .map(|row| {
                let client = client.clone();
                let cache = Arc::clone(&cache);
                async move {
                    let (pub_s, jour) =
                        resolve_labels(&row.paper_id, &row.details, &client, &cache).await;
                    (row.paper_id, row.status, row.message, pub_s, jour)
                }
            })
            .buffer_unordered(conc)
            .collect()
            .await
    };

    if let Some(ref p) = args.cache {
        let guard = cache.lock().await;
        save_cache(p, &*guard)?;
    }

    let mut by_group: HashMap<String, u64> = HashMap::new();
    let mut by_pub: HashMap<String, u64> = HashMap::new();
    let mut by_journal: HashMap<String, u64> = HashMap::new();

    for (paper_id, _status, _msg, pub_s, jour) in &resolved {
        let pub_e = if pub_s.is_empty() { "(empty)" } else { pub_s.as_str() };
        let jour_e = if jour.is_empty() { "(empty)" } else { jour.as_str() };
        *by_pub.entry(pub_e.to_string()).or_insert(0) += 1;
        *by_journal.entry(jour_e.to_string()).or_insert(0) += 1;
        let group_key = if !pub_s.trim().is_empty() {
            pub_s.clone()
        } else if !jour.trim().is_empty() {
            jour.clone()
        } else if !paper_id.is_empty() {
            paper_id.clone()
        } else {
            "(no paper_id)".to_string()
        };
        *by_group.entry(group_key).or_insert(0) += 1;
    }

    println!("files: {}", jsonl_files.len());
    println!("records_total: {}", total);
    println!("text_retrieval_failed: {}", n_failed);
    if total > 0 {
        println!("fraction_failed: {:.4}", n_failed as f64 / total as f64);
    }

    println!("\n# By primary group key (publisher, else journal, else paper_id)");
    let mut bg: Vec<_> = by_group.into_iter().collect();
    bg.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (k, v) in bg {
        println!("{}\t{}", v, k);
    }

    println!("\n# Publisher field");
    let mut bp: Vec<_> = by_pub.into_iter().collect();
    bp.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (k, v) in bp {
        println!("{}\t{}", v, k);
    }

    println!("\n# Journal title");
    let mut bj: Vec<_> = by_journal.into_iter().collect();
    bj.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    for (k, v) in bj {
        println!("{}\t{}", v, k);
    }

    if args.list_failed {
        println!("\n# failed paper_id\tstatus\tmessage\tgroup\tpublisher\tjournal");
        for (paper_id, status, msg, pub_s, jour) in &resolved {
            let group_key = if !pub_s.trim().is_empty() {
                pub_s.as_str()
            } else if !jour.trim().is_empty() {
                jour.as_str()
            } else {
                paper_id.as_str()
            };
            println!(
                "{}\t{}\t{}\t{}\tpublisher={}\tjournal={}",
                paper_id, status, msg, group_key, pub_s, jour
            );
        }
    }

    Ok(())
}
