from pytube import YouTube

yt = YouTube("https://www.youtube.com/watch?v=6Nt27AswquQ&ab_channel=Tinht%E1%BA%BF")
yt = yt.streams.first()
yt.download("/Users/ntdat/Downloads/DownloadFromYoutube/")
