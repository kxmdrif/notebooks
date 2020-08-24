# Math


## 1. Two Sum

**solution**

hashmap

- Time Complexity: O(n)

```java
import java.util.*;
class Solution {
   
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> m = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
            if (m.containsKey(target - nums[i]))
                return new int[] {m.get(target - nums[i]), i};
            m.put(nums[i], i);
        }
        return null;
    }
}
```

## 7. Reverse Integer
**slolution**
- Time Complexity: O(log(x)). There are roughly log10(x) digits in x
- Space Complexity: O(1)

```java
class Solution {
    public int reverse(int x) {
        int res = 0;
        while(x != 0){
            int r = x % 10;
            x = x / 10;
            if (res > Integer.MAX_VALUE / 10 || res == Integer.MAX_VALUE && r > 7)
                return 0;
            if (res < Integer.MIN_VALUE / 10 || res == Integer.MIN_VALUE && r < -8)
                return 0;
            res = 10 * res + r;
        }
        return res;
    }
}
```

## 15. 3Sum
**solution**
- Time Complexity: O(n^2)

```java
import java.util.*;
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length <= 2)
            return res;
        Arrays.sort(nums);
        
        for(int i = 0; i < nums.length - 2; i++){
            if (nums[i] > 0) return res;
            if (i > 0 && nums[i] == nums[i - 1]) 
                continue;
            int j = i + 1, k = nums.length - 1;
            int sum = -nums[i];
            while(j < k){
                if (nums[j] + nums[k] == sum){
                    res.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    while(j < k && nums[j] == nums[j + 1]) j++;
                    while(j < k && nums[k] == nums[k - 1]) k--;
                    j++;
                    k--;
                }else if (nums[j] + nums[k] > sum){
                    while(j < k && nums[k] == nums[k - 1]) k--;
                    k--;
                }else{
                    while(j < k && nums[j] == nums[j + 1]) j++;
                    j++;
                }
                    
            }
        }
        return res;
    }
}
```
