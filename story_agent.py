def generate_simple_story(raw_story: str, story_type: str) -> str:
    return f"""
Cerita ini adalah bagian dari kisah hidup yang berharga.

Bapak/Ibu menceritakan: "{raw_story}"

Dari cerita ini tampak bahwa hidup Bapak/Ibu tidak hanya terdiri dari kejadian biasa,
tetapi juga mengandung makna. Ada pengalaman, perasaan, relasi, dan nilai yang dapat
menjadi bagian dari kisah hidup yang lebih besar.

Cerita ini dapat disimpan sebagai bagian dari bab: {story_type}.
"""

def detect_values(raw_story: str):
    text = raw_story.lower()
    values = []

    if "anak" in text or "cucu" in text or "keluarga" in text:
        values.append("keluarga")
    if "doa" in text or "tuhan" in text or "gereja" in text:
        values.append("iman")
    if "kerja" in text or "mengajar" in text:
        values.append("kontribusi")
    if "sakit" in text or "sulit" in text or "sedih" in text:
        values.append("ketahanan")
    if "senang" in text or "syukur" in text:
        values.append("syukur")

    return values or ["makna hidup"]

def recommend_prompt(values):
    if "keluarga" in values:
        return "Apakah Bapak/Ibu ingin menyimpan cerita ini untuk anak atau cucu?"
    if "iman" in values:
        return "Apakah Bapak/Ibu ingin menambahkan doa pendek pada cerita ini?"
    if "kontribusi" in values:
        return "Apakah Bapak/Ibu ingin mengubah cerita ini menjadi nasihat untuk generasi muda?"
    return "Apakah Bapak/Ibu ingin melanjutkan cerita ini besok?"