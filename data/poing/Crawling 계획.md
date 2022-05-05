# Crawling 계획

> 곽현우: `naver map`
>
> 이가은님:  `Kakao Map`
>
> 이예준님:  `식신`
>
> 정동인님: `망고 플레이트`

- **검색어**: **( )구 + 음식점**
- **검색조건**: :star:이 하나인 별점이 최소 1만개가 될 때까지 크롤링

- **Data** **형식**

`pd.DataFrame`

| Region | Rating | Category | Review |
| ------ | ------ | -------- | ------ |
| ...    | ...    | ...      | ...    |

이 후 `parquet`으로 추출



- 다음 회의까지 Data 크롤링에 집중
- 이 후 회의에서 감정 분석(classification)관련 Task 진행에 관한 역할 분배 및 논의 필요

