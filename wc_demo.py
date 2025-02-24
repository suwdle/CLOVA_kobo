from wordcloud import WordCloud
import matplotlib.pyplot as plt


def generate_wordcloud(data): 

    text = " ".join(data)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'  # 한글 표시를 위한 폰트 경로 (OS에 따라 변경)
    ).generate(text)

    # 워드 클라우드 출력
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
