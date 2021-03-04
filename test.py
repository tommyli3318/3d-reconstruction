
def normalize(arr, a, b) -> None:
    # normalizes arr to be in range of (a, b)
    A = min(arr) # old min
    B = max(arr) # old max
    
    for i in range(len(arr)):
        arr[i] = round(a + (arr[i]-A)*(b-a) / (B-A), 2)



arr = [10,20,35,40,50]

normalize(arr, 1,5)

print(arr)