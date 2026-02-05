from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    score = analyzer.polarity_scores(text)['compound']
    return score

# Example usage
if __name__ == "__main__":
    sample = "Apple stock surges after strong earnings report!"
    print(f"Sentiment Score: {get_sentiment_score(sample)}")
