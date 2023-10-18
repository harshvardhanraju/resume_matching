from pypdf import PdfReader

reader = PdfReader(r'g:\jd_Cv\Profiles-20231016T064917Z-001\Profiles\Application Security Specialist\Ganesh_s Resume.pdf')
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()

print(text)