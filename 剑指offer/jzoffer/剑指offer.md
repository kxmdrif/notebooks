# 剑指offer

## 3. 数组中重复的数字
在一个长度为 n 的数组里的所有数字都在 0 到 n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字是重复的，也不知道每个数字重复几次(可能没有重复的)。请找出数组中任意一个重复的数字。

相似问题：长度为n + 1的数组，数字范围1 ~n, 只有一个重复但不知道重复几次，方法：快慢指针(从第一个元素开始)

```
Input:
{2, 3, 1, 0, 2, 5}
Output:
2
```

```java
/*也可以这样
    public boolean duplicate(int numbers[],int length,int [] duplication) {
        for(int i = 0; i < length; i++){
            while(numbers[i] != numbers[numbers[i]]){
                swap(numbers, i, numbers[i]);
            }
            if (numbers[i] != i){
                duplication[0] = numbers[i];
                return true;
            }
        }
        return false;
    }
*/
    public boolean duplicate(int[] nums, int length, int[] duplication) {
        if (nums == null || length <= 0)
            return false;
        for (int i = 0; i < length; i++) {
            while (nums[i] != i) {
                if (nums[i] == nums[nums[i]]) {
                    duplication[0] = nums[i];
                    return true;
                }
                swap(nums, i, nums[i]);
            }
        }
        return false;
    }
    private void swap(int[] nums, int i, int j) {
        int t = nums[i];
        nums[i] = nums[j];
        nums[j] = t;
    }
```
