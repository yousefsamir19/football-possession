from backend.football_analyzer import track_video


input_video = "inputs/football.mp4"
output_video = "outputs/tracked3.mp4"

track_video(
    input_video,
    output_video
)

print("Done!")
print(f"Output: {output_video}")