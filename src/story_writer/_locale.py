def translate(term, language='en'):

    if language == 'en':
        return term

    all_terms_en_de = {
        "AI Story writer":"KI Geschichtenschreiber",
        "Let AI write you a story.":"Die KI schreibt Dir eine Geschichte.",
        "Short story content:":"Kurzinhalt",
        "Story length (in sentences):":"Anzahl Saetze",
        "Target audience:":"Zielgruppe",
        "young adults":"junge Erwachsene",
        "Language:":"Sprache",
        "Generate image":"Generiere ein Bild",
        "Explain how it's made":"Erklaere wie es funktioniert",
        "Generate story":"Generiere Geschichte",
        "Advanced options":"Erweiterte Optionen",
        "Text generation model:":"Text-Generierungs-Model",
        "Image generation model:":"Bild-Generierungs-Model",
        "Random number:":"Zufaellige Zahl",
        "Disclaimer: This story has been auto-generated using artificial intelligence. Any resemblance to real persons, living or dead, or actual places or events is purely coincidental and unintentional. Read the documentation of the story-writer Python library to learn more: https://github.com/haesleinhuepf/story-writer":
        "Hinweis: Diese Geschichte wurde durch kuenstliche Intelligenz automatisch generiert. Jeder Zusammenhang mit realen Personen, tot oder lebendig, realen Orten oder Veranstaltungen sind rein zufaellig und nicht beabsichtigt. Lesen Sie die Dokumnetation der story-writer Python-Bibliothek um mehr zu erfahren: https://github.com/haesleinhuepf/story-writer",
        "Download Story":"Geschichte runterladen",
        "Randomize": "Zufall"
    }

    if term in all_terms_en_de.keys():
        return all_terms_en_de[term]

    return term
