"""Export compact report tables as Word-friendly HTML, booktabs LaTeX and CSV."""
from html import escape
from pathlib import Path


def _latex(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
                    '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}'}
    return ''.join(replacements.get(char, char) for char in str(value))


def export_report_tables(tables, output_dir, stem):
    """Use already formatted frames so notebook, HTML and LaTeX agree exactly.

    Open the HTML in a browser, select a table and paste into Word using source
    formatting. The LaTeX version requires the booktabs package.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html = ['<!doctype html><html><head><meta charset="utf-8"><style>',
            'body{font:11pt "Times New Roman",serif;max-width:960px;margin:32px auto;}',
            'table{border-collapse:collapse;width:100%;margin:8px 0;}',
            'th,td{padding:4px 7px;text-align:right;}th:first-child,td:first-child{text-align:left;}',
            'thead{border-top:1.5px solid black;border-bottom:1px solid black;}',
            'tbody{border-bottom:1.5px solid black;}caption{text-align:left;font-weight:bold;}',
            '.note{font-size:10pt;}section{margin-bottom:24px;}',
            '</style></head><body>']
    latex = ['% Requires \\usepackage{booktabs}']
    for number, item in enumerate(tables, 1):
        # Pivot column-axis names otherwise create a spurious HTML header cell.
        frame = item['frame'].rename_axis(None, axis=1)
        title, note, prose = (item.get(key, '') for key in ('title', 'note', 'prose'))
        html.extend(['<section>', '<p><b>' + escape(title) + '</b></p>',
                     frame.to_html(index=False, border=0, escape=True),
                     '<p class="note">' + escape(note) + '</p>',
                     '<p>' + escape(prose) + '</p></section>'])
        alignment = 'l' + 'r' * (len(frame.columns) - 1)
        latex.extend([r'\begin{table}[htbp]', r'\centering', r'\small',
                      r'\caption{' + _latex(title) + '}',
                      r'\begin{tabular}{' + alignment + '}', r'\toprule',
                      ' & '.join(map(_latex, frame.columns)) + r' \\', r'\midrule'])
        latex.extend(' & '.join(map(_latex, row)) + r' \\'
                     for row in frame.itertuples(index=False, name=None))
        latex.extend([r'\bottomrule', r'\end{tabular}',
                      r'\par\smallskip\begin{minipage}{\linewidth}\footnotesize '
                      + _latex(note) + r'\end{minipage}', r'\end{table}',
                      _latex(prose), ''])
        frame.to_csv(output_dir / f'{stem}_{number}.csv', index=False)
    html.append('</body></html>')
    paths = {'html': output_dir / f'{stem}.html', 'latex': output_dir / f'{stem}.tex'}
    paths['html'].write_text('\n'.join(html), encoding='utf-8')
    paths['latex'].write_text('\n'.join(latex), encoding='utf-8')
    return paths
