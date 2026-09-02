// MyST plugin: undo linkify's false positives on dotted Python identifiers.
//
// mystmd runs markdown-it's linkify unconditionally (no myst.yml switch), and
// linkify treats any `word.tld` as a URL when `tld` is a real top-level domain.
// `fit`, `data`, `save`, `io` are all TLDs, so prose like `BrainData.fit` or
// `self.data` renders as a dead `http://BrainData.fit` link. Docstrings and
// commit messages (the changelog) can't always wrap these in backticks, so this
// transform turns such links back into plain text.
//
// A link is treated as a false positive when linkify (not the author) supplied
// the scheme -- i.e. the url is exactly `http://` + the link text -- and the
// text looks like an attribute access rather than a hostname: it contains an
// uppercase letter or underscore, or its last segment is one of the TLDs that
// double as common Python attribute names.

const CODE_TLDS = new Set(["fit", "data", "save", "io", "cv", "py", "md", "so", "sh", "info", "name", "id"]);

function isIdentifierLink(node) {
  const text = node.children?.length === 1 ? node.children[0].value : undefined;
  if (!text || node.url !== `http://${text}`) return false;
  if (!/^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$/.test(text)) return false;
  if (/[A-Z_]/.test(text)) return true;
  return CODE_TLDS.has(text.split(".").pop());
}

const unlinkifyTransform = {
  name: "unlinkify-identifiers",
  doc: "Turn linkify's false-positive links on dotted Python identifiers back into text.",
  stage: "document",
  plugin: (_opts, utils) => (tree) => {
    for (const node of utils.selectAll("link", tree)) {
      if (!isIdentifierLink(node)) continue;
      node.type = "text";
      node.value = node.children[0].value;
      delete node.children;
      delete node.url;
      delete node.urlSource;
    }
  },
};

const plugin = { name: "nltools unlinkify", transforms: [unlinkifyTransform] };

export default plugin;
