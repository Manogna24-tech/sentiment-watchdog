from transformers import pipeline

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def analyze_emotion(text):
    raw_result = classifier(text)[0]
    emotion = raw_result['label']
    confidence = raw_result['score']
    normalized_score = round(confidence * 100, 2)

    return {
        "emotion": emotion,
        "confidence": normalized_score,
        "explanation": f"This text expresses {emotion} with {normalized_score}% confidence."
    }