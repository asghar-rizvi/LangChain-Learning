from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def extract_transcript_from_video_id(video_id):
    try:
        # transcript_list =  YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        transcript = " ".join(chunk.text for chunk in fetched_transcript)
        return transcript
    except TranscriptsDisabled:
        print("No captions available for this video")
if __name__ == '__main__':
    video_id = "0jspaMLxBig"
    transcript = extract_transcript_from_video_id(video_id=video_id)
    print(transcript[:1000])