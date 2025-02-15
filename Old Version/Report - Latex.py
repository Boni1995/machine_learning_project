from pylatex import Document, Section, Subsection, Command
from pylatex.utils import italic, NoEscape

def generar_reporte_pdf():
    # Crear un documento LaTeX básico
    doc = Document()

    # Título, autor y fecha
    doc.preamble.append(Command('title', 'Mi Primer Reporte en PDF'))
    doc.preamble.append(Command('author', 'Juan Pérez'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))

    # Crear la sección de introducción
    with doc.create(Section('Introducción')):
        doc.append('Este es un ejemplo de reporte generado directamente como PDF.')
        doc.append(italic('Esto es un texto en cursiva.'))
        doc.append('\nAdemás, puedes agregar más contenido como tablas, gráficos y secciones.')

    # Crear una subsección con información adicional
    with doc.create(Subsection('Datos Importantes')):
        doc.append('Aquí puedes incluir tablas, gráficos, o incluso ecuaciones matemáticas como esta:')
        doc.append(NoEscape(r'\[ E = mc^2 \]'))

    # Generar el archivo PDF directamente
    doc.generate_pdf('reporte_pdf', clean_tex=False)  # Crea el PDF y también guarda el archivo .tex
    print("PDF generado con éxito: reporte_pdf.pdf")

generar_reporte_pdf()