# 가설 5 — 유명도(사전학습 지식) 편향 분석

Bridge-hop only인데도 EM=1(암기 없이는 불가능 - bridge_hop_text에는 정답 문자열이 없도록 이미 필터링됨): 22/4734건

| 그룹 | n | 평균 fame | 중앙값 fame |
|---|---|---|---|
| EM=1(사실상 암기) | 22 | 2.27 | 1.0 |
| 나머지 | 4712 | 3.24 | 2.0 |
| 오답·타입 일치 | 3274 | 3.36 | - |
| 오답·타입 불일치 | 1438 | 2.96 | - |

EM=1 그룹의 fame이 나머지보다 뚜렷이 높다면 → 문맥 없이 정답을 맞힌 사례가 무작위가 아니라 유명 엔티티에 편중된다는 뜻(가설 5 지지).

## 케이스 스터디 — bridge_only EM=1 (암기로 추정되는 사례)

- qid=5ab7c24e5542995dae37e998 bridge_entity='Meuse-Argonne Offensive' (fame=12) Q: In which war was the Meuse-Argonne Offensive executed? answer='World War I.' conf=0.307
- qid=5ab2d1c45542992953946877 bridge_entity='Swanson' (fame=5) Q: As an American producer of canned soups and related products based in Camden, New Jersey, what company handles the broth business of the Swanson brand? answer='The Campbell Soup Company' conf=0.512
- qid=5a8492ab5542992a431d1a5b bridge_entity='Northrop F-15 Reporter' (fame=4) Q: Which plane has seen more combat, the Northrop F-15 Reporter or the Northrop P-61 Black Widow? answer='The Northrop P-61 Black Widow' conf=0.202
- qid=5ab48af35542996a3a969f92 bridge_entity='Fennec fox' (fame=4) Q: Which small nocturnal fox found in the Sahara of North Africa does the French animated series Fennec have a character? answer='Fennec fox' conf=0.896
- qid=5ab7f0015542992aa3b8c88b bridge_entity='Sean Yseult' (fame=3) Q: Who did the Star and Dagger bass player marry? answer='Sean Yseult.' conf=0.760
- qid=5a877fa45542993e715abf84 bridge_entity='Nana Patekar' (fame=3) Q: What actor born in 1951 starred in Aaj Ka Robin Hood? answer='Nana Patekar' conf=0.601
- qid=5ab3d69255429969a97a81c9 bridge_entity='USS Essex (CV-9)' (fame=2) Q: To which aircraft carrier was the VMF-213 Marine Fighting Squadron assigned to durning World War II? answer='USS Essex' conf=0.714
- qid=5ae494575542995dadf2434d bridge_entity='Jayaprakash Narayan' (fame=2) Q: Jayaprakash Narayan was posthumously awarded what highest Indian civilian award? answer='The Bharat Ratna' conf=0.957
- qid=5a820897554299676cceb1f4 bridge_entity='Murmur (album)' (fame=2) Q: Which founding member drew critical acclaim as a bass guitarist with melodic basslines  on Murmur? answer='Mike" Mills' conf=0.997
- qid=5ab7bbbf55429928e1fe38bf bridge_entity='Kansas City crime family' (fame=1) Q:  William Cammisano was part of which Mafia family? answer='Kansas City crime family' conf=0.656