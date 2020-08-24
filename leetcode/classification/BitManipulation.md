# BitManipulation

## 136. Single Number
**solution**
Approach: XOR

```java
class Solution {
    public int singleNumber(int[] nums) {
        int res = 0;
        for(int num : nums)
            res ^= num;
        return res;
    }
}
```

## 338. Counting Bits
**solution**

- Time Complexity: O(n)
- Space Complexity: O(n)

```java
/*
Explaination.
Take number X for example, 10011001.
Divide it in 2 parts:
<1>the last digit ( 1 or 0, which is " i&1 ", equivalent to " i%2 " )
<2>the other digits ( the number of 1, which is " f[i >> 1] ", equivalent to " f[i/2] " )
*/
class Solution {
    public int[] countBits(int num) {
        int[] res = new int[num + 1];
        res[0] = 0;
        for(int i = 1; i < num + 1; i++)
            res[i] = res[i >> 1] + (i & 1);
        return res;
    }
}
```
