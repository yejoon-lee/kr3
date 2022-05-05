# Web Crawling

We crawled the restaurant reviews from multiple websites.
- [Diningcode](https://www.diningcode.com/)
- [Kakao Map](https://map.kakao.com/)
- [Mangoplate](http://www.mangoplate.com/)
- [Poing](https://m.poing.io/)
- [Siksin](https://www.siksinhot.com/)

# Class Construction Criteria

3-class construction criteria for each source of data.  
Criteria is decided by us(human) after examining the actual review data.

|            | 0<br>(Negative) | 1<br>(Positive) | 2<br>(Ambiguous) |
|:----------:|:---------------:|:---------------:|:----------------:|
| diningcode |      0,1,2      |       4,5       |         3        |
|  kakaomap  |       1,2       |       4,5       |         3        |
| mangoplate |       별로      |       맛있다!    |       괜찮다      |
|    poing   |       0,1       |       4,5       |        2,3       |
|   siksin   |   1.0,1.5,2.0   |     4.5,5.0     |  2.5,3.0,3.5,4.0 |

# Proportion of Sources

Number of samples for each source in `kr3_raw`.

|            | 0<br>(Negative) | 1<br>(Positive) | 2<br>(Ambiguous) |
|:----------:|:---------------:|:---------------:|:----------------:|
| diningcode |       6024      |      84328      |       16288      |
|  kakaomap  |      44844      |      103471     |       19795      |
| mangoplate |      18504      |      161424     |       48134      |
|    poing   |       373       |      20479      |       3818       |
|   siksin   |       2560      |      25041      |      100208      |
|    Total   |      71066      |      388823     |      182957      |

# Preprocessing

- Eliminated emojis('😀'), escape sequences('\n', '\b'), speical characters('~','&'), 한글 자모('ㅋ','ㅠ')
- Used spell checker
- Limited length (`2 < len(review) < 4001`)

> These preprocessing deicisons are based on the vocabulary of pre-trained model we used. For exmaple, keeping 'ㅋ' might be a good strategy in other case where 'ㅋ' is included in the vocabulary.

> `kr3_raw` hadn't gone through any of the preprocessing mentioned above.

# Other columns ('Region' and 'Category')

We crawled *'Region'* as the region where the restaurant is located, and *'Category'* as the category of food. We want to warn the usage of *'Region'* because they are often the querys we used in the crawling process, instead of the real crawled information.