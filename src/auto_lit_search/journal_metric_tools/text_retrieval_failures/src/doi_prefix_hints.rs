//! Heuristic publisher labels from DOI prefix when Crossref / Europe PMC return nothing.
//! Does not replace APIs when they succeed; longest matching prefix wins.

const PREFIXES: &[(&str, &str)] = &[
    ("10.1128/", "American Society for Microbiology"),
    ("10.1126/", "American Association for the Advancement of Science"),
    ("10.1073/", "Proceedings of the National Academy of Sciences"),
    ("10.1038/", "Springer Nature"),
    ("10.1007/", "Springer"),
    ("10.1186/", "Springer Nature"),
    ("10.1016/", "Elsevier"),
    ("10.1002/", "Wiley"),
    ("10.1111/", "Wiley"),
    ("10.1021/", "American Chemical Society"),
    ("10.1093/", "Oxford University Press"),
    ("10.1017/", "Cambridge University Press"),
    ("10.1080/", "Taylor & Francis"),
    ("10.1371/", "Public Library of Science"),
    ("10.3389/", "Frontiers"),
    ("10.7554/", "eLife Sciences Publications"),
    ("10.1101/", "Cold Spring Harbor Laboratory"),
    ("10.1136/", "BMJ"),
    ("10.1109/", "IEEE"),
    ("10.1145/", "Association for Computing Machinery"),
    ("10.5194/", "Copernicus Publications"),
    ("10.2147/", "Dove Medical Press"),
];

pub fn publisher_for_doi(doi: &str) -> Option<&'static str> {
    let d = doi.trim().to_ascii_lowercase();
    let mut best: Option<&'static str> = None;
    let mut best_len = 0usize;
    for (prefix, name) in PREFIXES {
        if d.starts_with(prefix) && prefix.len() >= best_len {
            best = Some(*name);
            best_len = prefix.len();
        }
    }
    best
}
