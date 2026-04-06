use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(about = "Scan collect/docling JSONL; count text retrieval failures by publisher/journal")]
pub struct Args {
    #[arg(long, num_args = 0..)]
    pub manifest: Vec<PathBuf>,
    #[arg(long)]
    pub glob: Option<String>,
    #[arg(long)]
    pub root: Option<PathBuf>,
    #[arg(
        long,
        env = "AUTO_LIT_PAPERS_DIR",
        help = "Recursively scan for *.jsonl (e.g. auto_lit_eval_data/papers)"
    )]
    pub papers_dir: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    pub recursive: bool,
    #[arg(long, default_value_t = false)]
    pub require_existing_file: bool,
    #[arg(long, default_value_t = 32u32, help = "Concurrent HTTP metadata requests")]
    pub concurrency: u32,
    #[arg(long, default_value_t = false)]
    pub no_resolve: bool,
    #[arg(long)]
    pub cache: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    pub list_failed: bool,
}

pub fn collect_jsonl_paths(args: &Args) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = Vec::new();
    paths.extend(args.manifest.iter().cloned());
    if let Some(ref g) = args.glob {
        for e in glob::glob(g).with_context(|| format!("bad glob {}", g))? {
            paths.push(e?);
        }
    }
    if let Some(ref r) = args.root {
        if args.recursive {
            for e in WalkDir::new(r).into_iter().filter_map(|e| e.ok()) {
                if e.file_type().is_file() {
                    let p = e.path();
                    if p.extension().map(|x| x == "jsonl").unwrap_or(false) {
                        paths.push(p.to_path_buf());
                    }
                }
            }
        } else {
            for e in fs::read_dir(r).with_context(|| format!("read_dir {}", r.display()))? {
                let e = e?;
                let p = e.path();
                if p.extension().map(|x| x == "jsonl").unwrap_or(false) {
                    paths.push(p);
                }
            }
        }
    }
    if let Some(ref pd) = args.papers_dir {
        for e in WalkDir::new(pd).into_iter().filter_map(|e| e.ok()) {
            if e.file_type().is_file() {
                let p = e.path();
                if p.extension().map(|x| x == "jsonl").unwrap_or(false) {
                    paths.push(p.to_path_buf());
                }
            }
        }
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}
