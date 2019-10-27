# Graduation

python version: 3.6.5, tensorflow version: 1.13.1

바탕화면에 엑셀파일들을 깔아놓고 exercise.py를 구동하면 바탕화면에 result.xls파일을 생성할 것입니다. 이 파일이 예측값입니다.

아직 코드가 완성되진 않았는데 예측값을 내야하는 상황이라 러프한 상태입니다. multi-task learning 단계는 아니고 2 inputs & 1 output인 상황입니다.


*19-10-28 추가*

 엑셀파일들 중 kpx 파일과 끝에 b가 붙은 파일들이 조금씩 수정되었습니다. kpx 파일에는 kpx sum 열이 추가되었고, b 파일들은 풍향을 rad로 변환한 뒤 cosθ+sinθ를 구하는 식과 년-월-일의 data 형식을 오직 일로만 표현하는 열이 추가되었습니다.
 
 exercise2 파일에 multi-task-learning을 구현한 코드가 존재하고, input data 종류가 풍속, 풍향에서 날짜 값이 추가되어 3-dimension을 이용하는 모델로 수정되었습니다.
