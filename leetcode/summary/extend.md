# extend

## 1. 最长递增子序列(求最长的那个子序列，若有多个，求字典序最小的)

```java
public class Solution {
    /**
     * retrun the longest increasing subsequence
     * @param arr int整型一维数组 the array
     * @return int整型一维数组
     */
    public int[] LIS (int[] arr) {
        // tails[i] ==> 长度为i + 1的递增子序列的最小结尾值
        int[] tails = new int[arr.length];
        //dp[i] ==> 以arr[i]结尾的最长子序列长度
        int[] dp = new int[arr.length];
        
        int size = 0;
        for(int i = 0; i < arr.length; i++){
            int l = 0, r = size;
            while(l < r){
                int mid = l + ((r - l)>>1);
                if (tails[mid] < arr[i])
                    l = mid + 1;
                else
                    r = mid;
            }
            tails[l] = arr[i];
            dp[i] = l + 1;
            if (l == size){
                ++size;
            }
            
        }
        int[] res = new int[size];
        for(int i = arr.length - 1; i >= 0; i--){
            if(dp[i] == size)
                res[--size] = arr[i];
        }
        return res;
    }
}
```