from pydub import AudioSegment

filepath = r"D:\Program Files\JiJiDown\Download\李宏毅2020机器学习深度学习(完整版)国语 - 3.Regression - Case Study(Av94519857,P3).mp3"
savepath = r"C:\Users\msi-pc\Desktop\Voices\Men"
mp3_file = AudioSegment.from_mp3(filepath) # 读取mp3文件
start = 100000 #从100s开始
end = 3000000 #到3000s结束
file_to_cut = mp3_file[start:end]
cut_start = start
count = 0
while cut_start < end:
    cut_end = cut_start + 5000 #一段音频的长度是5秒
    output = file_to_cut[cut_start:cut_end]
    output.export(savepath+"\\Men_voice%d.flac"%count, format="flac")
    cut_start = cut_end
    count += 1
    if count > 499:
        break

filepath = r"D:\Program Files\JiJiDown\Download\懒猫老师-C语言-链表（单链表，循环链表） - 1.链表剪辑降噪(Av77564690,P1).mp3"
savepath = r"C:\Users\msi-pc\Desktop\Voices\Women"
mp3_file = AudioSegment.from_mp3(filepath) # 读取mp3文件
start = 100000 #从100s开始
end = 3000000 #到3000s结束
file_to_cut = mp3_file[start:end]
cut_start = start
count = 0
while cut_start < end:
    cut_end = cut_start + 5000 #一段音频的长度是5秒
    output = file_to_cut[cut_start:cut_end]
    output.export(savepath+"\\Women_voice%d.flac"%count, format="flac")
    cut_start = cut_end
    count += 1
    if count > 499:
        break
