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