# BinarySraech

## 33. Search in Rotated Sorted Array
**solution**
Approach: Binary search

- Time complexity: O(log n)
- Space complexity: O(1)

```java
class Solution {
    public int search(int[] nums, int target) {
        return binarySearch(nums, 0, nums.length - 1, target);
    }
    private int binarySearch(int[] nums, int start, int end,
                             int target){
        while(start <= end){
            int mid = start + (end - start) / 2;
            if (nums[mid] == target)
                return mid;
            if (nums[mid] >= nums[start]){
                if (target >= nums[start] && target < nums[mid])
                    end = mid - 1;
                else
                    start = mid + 1;
            }else{
                if (target > nums[mid] && target <= nums[end])
                    start = mid + 1;
                else
                    end = mid - 1;
            }
        }
        return -1;
    }
}
```

## 34. Find First and Last Position of Element in Sorted Array 

**solution**

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] res = {-1, -1};
        int left = binarySearch(nums, target, true);
        int right = binarySearch(nums, target, false);
        res[0] = (left == nums.length || nums[left] != target) 
            ? -1 : left;
        res[1] = (right == 0 || nums[right - 1] != target)
            ? -1 : right - 1;
        return res;
    }
    //attention for nums is empty!
    private int binarySearch(int[] nums, int target, boolean isleft){
        int l = 0, r = nums.length;
        while(l < r){
            int mid = l + ((r - l) >> 1);
            if (target < nums[mid] || (isleft && nums[mid] == target))
                r = mid;
            else
                l = mid + 1;
        }
        return l;
    }
    
}
```

## 300. Longest Increasing Subsequence
**solution**

Approach1: DP

- Time Complexity: O(n^2)
- Space Complexity: O(n)

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        //dp[i]=max(dp[j])+1,∀0≤j<i and dp[i] >dp[j]
        int n = nums.length;
        if (n == 0) return 0;
        int[] dp = new int[n];
        dp[0] = 1;
        int maxlen = 1;//for example: [0]
        for(int i = 1; i < n; i++){
            int maxval = 0;
            for(int j = 0; j < i; j++)
                if (nums[i] > nums[j])
                    maxval = Math.max(maxval, dp[j]);
            dp[i] = maxval + 1;
            maxlen = Math.max(dp[i], maxlen);            
        }
        return maxlen;
        
    }
    
}
```

Approach 2 : Binary Search


- Time Complexity: O(nlogn)
- Space Complexity: O(n)

tails is an array storing the smallest tail of all increasing subsequences with length i+1 in tails[i].
For example, say we have nums = [4,5,6,3], then all the available increasing subsequences are:

len = 1   :      [4], [5], [6], [3]   => tails[0] = 3
len = 2   :      [4, 5], [5, 6]       => tails[1] = 5
len = 3   :      [4, 5, 6]            => tails[2] = 6
We can easily prove that tails is a increasing array. Therefore it is possible to do a binary search in tails array to find the one needs update.

Each time we only do one of the two:

(1) if x is larger than all tails, append it, increase the size by 1
(2) if tails[i-1] < x <= tails[i], update tails[i]

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] tails = new int[nums.length];
        int size = 0;
        for(int num : nums){
            int l = 0, r = size;
            while(l < r){
                int mid = l + ((r - l) >> 1);
                if (tails[mid] < num)
                    l = mid + 1;
                else
                    r = mid;
            }
            //if num exists l is the pos of num in tails
            //else l is the pos where tails[pos] is larger than num firstly
            tails[l] = num;
            if (l == size) ++size;
        }
        return size;
    }
}
```

Example: For [1,3,5,2,8,4,6]
For this list, we can have LIS with different length.
For length = 1, [1], [3], [5], [2], [8], [4], [6], we pick the one with smallest tail element as the representation of length=1, which is [1]
For length = 2, [1,2] [1,3] [3,5] [2,8], ...., we pick [1,2] as the representation of length=2.
Similarly, we can derive the sequence for length=3 and length=4
The result sequence would be:
len=1: [1]
len=2: [1,2]
len=3: [1,3,4]
len=4: [1,3,5,6]

According to the logic in the post,we can conclude that:
(1) If there comes another element, 9
We iterate all the sequences, found 9 is even greater than the tail of len=4 sequence, we then copy len=4 sequence to be a new sequece, and append 9 to the new sequence, which is len=5: [1,3,5,6,9]
The result is:
len=1: [1]
len=2: [1,2]
len=3: [1,3,4]
len=4: [1,3,5,6]
len=5: [1,3,5,6,9]

(2) If there comes another 3,
We found len=3 [1,3,4], whose tailer is just greater than 3, we update the len=3 sequence tobe [1,3,3]. The result is:
len=1: [1]
len=2: [1,2]
len=3: [1,3,3]
len=4: [1,3,5,6]

(3) If there comes another 0,
0 is smaller than the tail in len=1 sequence, so we update the len=1 sequence. The result is:
len=1: [0]
len=2: [1,2]
len=3: [1,3,3]
len=4: [1,3,5,6]

