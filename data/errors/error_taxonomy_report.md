# 에러 taxonomy(가설 1 핵심 결과)

Full 정답 2775건 중 shortcut_success 2515건 / bridge_needed 260건

## 그룹별 특징 비율

| 그룹 | n | type-constraint cue | qa_word_overlap 평균 | bridge 제목이 질문에 | bridge 제목이 answer_hop에 |
|---|---|---|---|---|---|
| shortcut_success | 2515 | 25.09% | 0.433 | 50.66% | 20.60% |
| bridge_needed | 260 | 11.92% | 0.424 | 26.54% | 56.15% |

## Bridge-hop only 오답 taxonomy

오답 4712건 중 타입 일치 69.48% (일치 시 평균 confidence 0.517, 불일치 시 0.309)

## 케이스 스터디 — shortcut_success (질문/문단만으로 충분)

- qid=5a85b2d95542997b5ce40028
  Q: Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?
  answer: Eenasul Fateh
  answer_hop(Eenasul Fateh): Eenasul Fateh (Bengali: ঈনাসুল ফাতেহ ; born 3 April 1959), also known by his stage name Aladin, is a Bangladeshi-British cultural practitioner, magician, live artist and former international managemen...
  bridge_hop(Management consulting): Management consulting is the practice of helping organizations to improve their performance, operating primarily through the analysis of existing organizational problems and the development of plans f...
  answer_only_pred: 'Eenasul Fateh' | bridge_only_pred: 'management consultants' (conf=0.002)
  type_constraint_cue=False qa_word_overlap=0.55 bridge_in_question=False bridge_in_answer_hop=False

- qid=5a7166395542994082a3e814
  Q: What is the name of the fight song of the university whose main campus is in Lawrence, Kansas and whose branch campuses are in the Kansas City metropolitan area?
  answer: Kansas Song
  answer_hop(Kansas Song): Kansas Song (We’re From Kansas) is a fight song of the University of Kansas....
  bridge_hop(University of Kansas): The University of Kansas, often referred to as KU or Kansas, is a public research university in the U.S. state of Kansas. The main campus in Lawrence, one of the largest college towns in Kansas, is on...
  answer_only_pred: 'Kansas Song' | bridge_only_pred: 'Edwards Campus in Overland Park' (conf=0.085)
  type_constraint_cue=False qa_word_overlap=0.29 bridge_in_question=False bridge_in_answer_hop=True

- qid=5a75e05c55429976ec32bc5f
  Q: Brown State Fishing Lake is in a country that has a population of how many inhabitants ?
  answer: 9,984
  answer_hop(Brown County, Kansas): Brown County (county code BR) is a county located in the northeast portion of the U.S. state of Kansas. As of the 2010 census, the county population was 9,984. Its county seat and most populous city i...
  bridge_hop(Brown State Fishing Lake): Brown State Fishing Lake (sometimes also known as Brown State Fishing Lake And Wildlife Area) is a protected area in Brown County, Kansas in the United States. The lake is 62 acres (0.25 km²) in area ...
  answer_only_pred: '9,984' | bridge_only_pred: 'United States' (conf=0.244)
  type_constraint_cue=True qa_word_overlap=0.38 bridge_in_question=True bridge_in_answer_hop=False

- qid=5ae0d4c9554299603e418468
  Q: Roger O. Egeberg was Assistant Secretary for Health and Scientific Affairs during the administration of a president that served during what years?
  answer: 1969 until 1974
  answer_hop(Richard Nixon): Richard Milhous Nixon (January 9, 1913 – April 22, 1994) was the 37th President of the United States from 1969 until 1974, when he resigned from office, the only U.S. president to do so. He had previo...
  bridge_hop(Roger O. Egeberg): Roger Olaf Egeberg, M.D. (13 November 1902 – 13 September 1997 Washington, D.C.) was an American medical educator, administrator and advocate of public health. He was General Douglas MacArthur's perso...
  answer_only_pred: '1969 until 1974' | bridge_only_pred: 'Nixon administration' (conf=0.342)
  type_constraint_cue=False qa_word_overlap=0.15 bridge_in_question=True bridge_in_answer_hop=False

## 케이스 스터디 — bridge_needed (bridge 없이는 실패)

- qid=5ae7a8175542993210983ed8
  Q: Which other Mexican Formula One race car driver has held the podium besides the Force India driver born in 1990?
  answer: Pedro Rodríguez
  answer_hop(Formula One drivers from Mexico): There have been six Formula One drivers from Mexico who have taken part in races since the championship began in 1950. Pedro Rodríguez is the most successful Mexican driver being the only one to have ...
  bridge_hop(Sergio Pérez): Sergio Pérez Mendoza (    ; born 26 January 1990) also known as "Checo" Pérez, is a Mexican racing driver, currently driving for Force India....
  answer_only_pred: 'Sergio Pérez' | bridge_only_pred: 'Sergio Pérez Mendoza' (conf=0.741)
  type_constraint_cue=False qa_word_overlap=0.57 bridge_in_question=False bridge_in_answer_hop=True

- qid=5abf63f15542997ec76fd3ea
  Q: Alexander Kerensky was defeated and destroyed by the Bolsheviks in the course of a civil war that ended when ?
  answer: October 1922
  answer_hop(Russian Civil War): The Russian Civil War (Russian: Гражда́нская война́ в Росси́и , "Grazhdanskaya voyna v Rossiyi" ; November 1917 – October 1922) was a multi-party war in the former Russian Empire immediately after the...
  bridge_hop(Socialist Revolutionary Party): The Socialist Revolutionary Party, or Party of Socialists-Revolutionaries (the SRs; Russian: Партия социалистов-революционеров (ПСР), эсеры , "esery") was a major political party in early 20th century...
  answer_only_pred: '1923' | bridge_only_pred: 'February Revolution of 1917' (conf=0.821)
  type_constraint_cue=True qa_word_overlap=0.56 bridge_in_question=False bridge_in_answer_hop=False

- qid=5a8e1027554299653c1aa15f
  Q: Which year and which conference was the 14th season for this conference as part of the NCAA Division that the Colorado Buffaloes played in with a record of 2-6 in conference play?
  answer: 2009 Big 12 Conference
  answer_hop(2009 Big 12 Conference football season): The 2009 Big 12 Conference football season was the 14th season for the Big 12, as part of the 2009 NCAA Division I FBS football season....
  bridge_hop(2009 Colorado Buffaloes football team): The 2009 Colorado Buffaloes football team represented the University of Colorado in the 2009 NCAA Division I FBS college football season. The Buffaloes were led by fourth year head coach Dan Hawkins a...
  answer_only_pred: '2009 Big 12 Conference football season' | bridge_only_pred: 'Big 12' (conf=0.762)
  type_constraint_cue=True qa_word_overlap=0.43 bridge_in_question=False bridge_in_answer_hop=False

- qid=5ae7e1fc55429952e35ea9cc
  Q: What color clothing do people of the Netherlands wear during Oranjegekte or to celebrate the national holiday Koningsdag? 
  answer: orange
  answer_hop(Oranjegekte): Oranjegekte (Orange craze) or Oranjekoorts (Orange fever) is a phenomenon in the Netherlands that occurs during major sporting events, especially international football championships, and during Konin...
  bridge_hop(Koningsdag): Koningsdag (] ) or King's Day is a national holiday in the Kingdom of the Netherlands. Celebrated on 27 April (26 April if the 27th is a Sunday), the date marks the birth of King Willem-Alexander....
  answer_only_pred: 'T-shirts' | bridge_only_pred: "King's Day" (conf=0.150)
  type_constraint_cue=False qa_word_overlap=0.64 bridge_in_question=True bridge_in_answer_hop=True

## 케이스 스터디 — bridge_only 오답 (타입 일치, 그럴듯한 오답)

- qid=5a80541d5542996402f6a4d2
  Q: In what year did the Guild of Music Supervisors Awards recognize a British-American romantic drama film based on the 1952 romance novel "The Price of Salt" starring Cate Blanchett?
  answer: 2016
  answer_hop(Guild of Music Supervisors Awards): The Guild of Music Supervisors Awards recognize music supervisors in 14 categories, representing movies, television, games and trailers. "Compton", "Carol" and "Furious 7" were among the winners of th...
  bridge_hop(Carol (film)): Carol is a 2015 British-American romantic drama film directed by Todd Haynes. The screenplay, written by Phyllis Nagy, is based on the 1952 romance novel "The Price of Salt" (also known as "Carol") by...
  answer_only_pred: '2016' | bridge_only_pred: '2015' (conf=1.000)
  type_constraint_cue=True qa_word_overlap=0.25 bridge_in_question=False bridge_in_answer_hop=False

- qid=5ae352285542994393b9e685
  Q: Rumble Fish was a novel by the author of the coming-of-age novel published in what year by Viking Press?
  answer: 1967
  answer_hop(The Outsiders (novel)): The Outsiders is a coming-of-age novel by S. E. Hinton, first published in 1967 by Viking Press. Hinton was 15 when she started writing the novel, but did most of the work when she was 16 and a junior...
  bridge_hop(Rumble Fish (novel)): Rumble Fish is a 1975 novel for young adults by S. E. Hinton, author of "The Outsiders". It was adapted to film and directed by Francis Ford Coppola in 1983....
  answer_only_pred: '1967' | bridge_only_pred: '1975' (conf=1.000)
  type_constraint_cue=True qa_word_overlap=0.70 bridge_in_question=False bridge_in_answer_hop=False

- qid=5ac4e07e5542996feb3fe95c
  Q: The Bragg–Gray cavity theory was developed by Louis Harold Gray, William Lawrence Bragg, and a man that was knighrted in what year?
  answer: 1920
  answer_hop(William Henry Bragg): Sir William Henry Bragg (2 July 1862 – 12 March 1942) was a British physicist, chemist, mathematician and active sportsman who uniquely shared a Nobel Prize with his son William Lawrence Bragg – the 1...
  bridge_hop(Bragg–Gray cavity theory): According to the Bragg–Gray cavity theory, the ionization produced in a small cavity within an irradiated medium or object is related to the energy absorbed in that medium as a result of its radiation...
  answer_only_pred: '1915' | bridge_only_pred: '1936' (conf=1.000)
  type_constraint_cue=True qa_word_overlap=0.25 bridge_in_question=True bridge_in_answer_hop=False

- qid=5ac5391e5542994611c8b439
  Q: Sal de Mi Piel is a song by the actress who is of what nationality?
  answer: Spanish
  answer_hop(Belinda Peregrín): Belinda Peregrín Schüll (born August 15, 1989), known mononymously as Belinda, is a Spanish singer and actress Mexican naturalized ....
  bridge_hop(Sal de Mi Piel): "Sal de Mi Piel" (English: "Get Out of My Skin"), is a song by famous Mexican actress and singer Belinda....
  answer_only_pred: 'Spanish' | bridge_only_pred: 'Mexican' (conf=1.000)
  type_constraint_cue=True qa_word_overlap=0.14 bridge_in_question=True bridge_in_answer_hop=False
