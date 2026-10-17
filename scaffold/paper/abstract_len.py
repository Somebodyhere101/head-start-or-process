"""Length of the abstract as plain text (arXiv limit: 1,920 characters).   python abstract_len.py"""
import io, re
s = io.open('main.tex', encoding='utf-8').read()
a = s[s.index(r'\begin{abstract}') + len(r'\begin{abstract}'):s.index(r'\end{abstract}')]
t = re.sub(r'\\emph\{([^}]*)\}', r'\1', a)
t = re.sub(r'\\text\{([^}]*)\}', r'\1', t)
t = t.replace('$', '').replace('\\', '').replace('{', '').replace('}', '').replace('~', ' ')
t = ' '.join(t.split())
print('abstract: %d characters, %d words (arXiv limit 1920 characters)' % (len(t), len(t.split())))
