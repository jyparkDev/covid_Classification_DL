data= [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

print(len(data))
k = 4
# //는 몫만을 구하는 나누기 연산
num_validation = len(data) // k
validation_scores = []

for fold in range(k):
    validation_data = data[num_validation * fold: num_validation * (fold + 1)]
    # 리스트 + 리스트는 연결된 하나의 리스트를 생성한다
    train_data = data[:(num_validation * fold) ]+data[num_validation *(fold+1):] 
    print(validation_data)   
    print(train_data)