from wordcloud import WordCloud
import matplotlib.pyplot as plt

'''
계속 업데이트 -> 비효율적.
시간이나 질문 개수 기준을 지정해 요건 충족시 호출해서 사용할 수 있도록 함수를 구현해야 함
input : questions
    how? : load from backend
        query and content만 일괄적으로 가져오기
        한 사용자 전용? 전체 사용자에 대해 적용?
        
output : wordcloud (frontend에서 파싱할 수 있도록 반환)
'''
def generate_wordcloud(data): 

    text = " ".join(data)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'  # 한글 표시를 위한 폰트 경로
    ).generate(text)

    # 워드 클라우드 출력
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
