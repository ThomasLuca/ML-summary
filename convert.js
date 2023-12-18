const fs = require('fs');
const MarkdownIt = require('markdown-it');
const katex = require('@iktakahiro/markdown-it-katex');

// Initialize Markdown-It with the LaTeX plugin
const md = new MarkdownIt().use(katex);

// Read the input Markdown file
const inputMarkdown = fs.readFileSync('./summary.md', 'utf8');

// Convert Markdown to HTML
const htmlOutput = md.render(inputMarkdown);

// Write the HTML output to a new file
fs.writeFileSync('output.html', htmlOutput, 'utf8');


