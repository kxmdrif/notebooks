# Array

## 4. Median of Two Sorted Arrays

**solution**
O(log n)

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        //for an array with length n, whether n is odd or even, 
        //the Median is always the average of the (n + 1)/2 th and the (n + 2)/2 th
        int left = (m + n + 1) / 2, right = (m + n + 2) / 2;
        return (findKth(nums1, nums2, 0, 0, left) + 
            findKth(nums1, nums2, 0, 0, right)) / 2.0;
    }
    
    // find kth smallest item in two mergered array
    public int findKth(int[] nums1, int[] nums2, int i, int j, int k){
        if (i >= nums1.length) return nums2[j + k - 1];
        if (j >= nums2.length) return nums1[i + k - 1];
        if (k == 1) return Math.min(nums1[i], nums2[j]);
        
        //if midVal1 is out of range, kth might be in nums1[i, )
        //but kth couldn't be in nums2[i, i + k / 2) 
        //so we use INT MAX to prevent nums1[i, ) from being elimilated
        int midVal1 = (i + k / 2 - 1) < nums1.length
            ? nums1[i + k / 2 - 1] : Integer.MAX_VALUE;
        int midVal2 = (j + k / 2 - 1) < nums2.length
            ? nums2[j + k / 2 - 1] : Integer.MAX_VALUE;
        
        return midVal1 < midVal2 
            ? findKth(nums1, nums2, i + k / 2, j, k - k / 2)
            : findKth(nums1, nums2, i, j + k / 2, k - k / 2);
        //k - k / 2 couldn't be replaced by k / 2!!!
    }
    
    
}
```

## 11. Container With Most Water
**solution**

Initially we consider the area constituting the exterior most lines. Now, to maximize the area, we need to consider the area between the lines of larger lengths. If we try to move the pointer at the longer line inwards, we won't gain any increase in area, since it is limited by the shorter line. But moving the shorter line's pointer could turn out to be beneficial, as per the same argument, despite the reduction in the width. This is done since a relatively longer line obtained by moving the shorter line's pointer might overcome the reduction in area caused by the width reduction.

- Time Complexity: O(n)
- Space Complexity: O(1)

```java
class Solution {
    public int maxArea(int[] height) {
        int res = 0;
        int l = 0, r = height.length - 1;
        while(l < r){
            res = Math.max(res, 
                    Math.min(height[l], height[r]) * (r - l));
            if(height[l] < height[r])
                l++;
            else
                r--;
        }
        return res;
    }
}
```

## 41. First Missing Positive
**solution**
- Time complexity: O(n)
- Space Complexity: O(1)

缺失的数字一定在[1,n+1]的范围中，n为数组长度+1遍历数组时，对于每一位要while循环至该位置的数字(1～n)放到正确的位置为止, 若(1~n)有重复则重复的只有一个在正确位置。

```java
class Solution {
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        /*
        while里面最后一个条件不能写: nums[i] = i + 1(检测i位置是否放置了该放的值)
        而应该写nums[nums[i] - 1] != nums[i](检测nums[i]应该放置的位置，
        即nums[i] - 1位置是否放置了该放的值)
        否则对于nums[i] = nums[nums[i]]的情况，如[1, 1], [2, 2]会造成死循环
        或者可以写成这样
        while(nums[i] >= 1 && nums[i] <= n 
                  && nums[i] - 1 != i + 1){
                if (nums[nums[i] - 1] == nums[i])
                    break;
                swap(nums, i, nums[i] - 1);
            }
        */
        for(int i = 0; i < n; i++){
            while(nums[i] >= 1 && nums[i] <= n 
                  && nums[nums[i] - 1] != nums[i])
                swap(nums, i, nums[i] - 1);
        }
        for(int i = 0; i < n; i++)
            if(nums[i] != i + 1)
                return i + 1;
        return n + 1;
    }
    
    private void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```

## 42. Trapping Rain Water

**solution**
Approach 1: Dynamic Programming

- Time complexity: O(n)
- Space complexity: O(n)

```java
class Solution {
    public int trap(int[] height) {
        
        int res = 0;
        int n = height.length;
        if (n < 3) return 0;
        
        int[] leftMax = new int[n];
        int[] rightMax = new int[n];
        
        leftMax[0] = height[0];
        for (int i= 1; i < n; i++){
            leftMax[i] = Math.max(leftMax[i - 1], height[i]);
        }
        
        rightMax[n - 1] = height[n - 1];
        for(int i = n - 2; i >= 0; i--){
            rightMax[i] = Math.max(rightMax[i + 1], height[i]);
        }
        
        for(int i = 1; i < n - 1; i++)
            res = res + Math.min(rightMax[i], leftMax[i]) - height[i];
        
        return res;
    }
}
```
Approach 2: Using 2 pointers:

- Time complexity: O(n)
- Space complexity: O(1)

```java
class Solution {
    public int trap(int[] height) {
        int leftmax = 0, rightmax = 0;
        int l = 0, r = height.length - 1;
        int res = 0;
        /*注意是l <= r而不是l < r否则某个位置没有计算
        比如r - l = 1时只计算了其中一个位置之后就结束循环，另一个位置没计算
        */
        while(l <= r){
            if (leftmax < rightmax){
                res += Math.max(0, leftmax - height[l]);
                leftmax = Math.max(leftmax, height[l++]);
            }else{
                res += Math.max(0, rightmax - height[r]);
                rightmax = Math.max(rightmax, height[r--]);
            }
        }
        return res;
    }
}
```

## 45. Jump Game II
**solution**

- Time complexity: O(n)
- Space complexity: O(1)

About i < nums.length - 1:
The for loop logic checks whether we need to jump back (to a certain previous position) in order to jump further when we are at position i . We don't need to check whether we need to jump further when we already at the last position A.Length-1.

```java
class Solution {
    public int jump(int[] nums) {
        //end is the farthest pos that curr pos can reach, maxPos is the farthest pos that [curr, end] can reach
        int minJump = 0, end = 0, maxPos = 0;
        //注意i < nums.length - 1: 以[2, 3, 1, 1, 4]为例
        for(int i = 0; i < nums.length - 1; i++){
            maxPos = Math.max(maxPos, i + nums[i]);
            if (i == end){
                minJump++;
                end = maxPos;
            }
        }
        return minJump;
        
    }
}
```

## 53. Maximum Subarray
**solution**

- Time complexity: O(n)
- Space complexity: O(1)

```java
class Solution {
    public int maxSubArray(int[] nums) {
    //maxSum must be one of the possible subarray sum(consider [-1])
        int thisSum = 0, maxSum = nums[0];
        for(int num : nums){
            thisSum += num;
            maxSum = Math.max(maxSum, thisSum);
            if (thisSum < 0)
                thisSum = 0;
        }
        return maxSum;
    }
}
```

## 55. Jump Game

**soultion**
Approach 1 : (Dynamic Programming Top-down)      O(n^2)

```java
class Solution {
    int[] mem;
    public boolean canJump(int[] nums) {
        mem = new int[nums.length];
        for(int i = 0; i < nums.length; i++) mem[i] = -1;
        return canJumpHelper(nums, 0);
    }
    public boolean canJumpHelper(int[] nums, int beginIndex){
        if(mem[beginIndex] == 0) return false;
        if(mem[beginIndex] == 1) return true;
        
        if (beginIndex >= nums.length - 1) return true;
        boolean res = false;
        for(int i = 1; i <= nums[beginIndex]; i++){
            int nextIndex = beginIndex + i;
            res = canJumpHelper(nums, nextIndex);
            if (res) return true;
        }
        mem[beginIndex] = res ? 1 : 0;
        return res;
    }
}
```



Approach 2:   greedy O(n)

```java
class Solution {
    public boolean canJump(int[] nums) {
        int pos = nums.length - 1;
        for(int i = nums.length - 1; i >= 0; i--)
            if (nums[i] + i >= pos)
                pos = i;
        return pos == 0;
    }
}
```

## 56. Merge Intervals
**solution**
Approach : Sorting

- Time complexity:  O(nlogn)
- Space complexity: O(1) or O(n)(depends on the sort algorithm)

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) return new int[0][0];
        Arrays.sort(intervals, (o1, o2) -> (o1[0] - o2[0]));
        int min = intervals[0][0];
        int max = intervals[0][1];
        List<int[]> reslist = new ArrayList<>();
        for(int i = 1; i < intervals.length; i++){
            if (intervals[i][0] <= max){
                max = Math.max(max, intervals[i][1]);
            }else{
                reslist.add(new int[]{min, max});
                min = intervals[i][0];
                max = intervals[i][1];
            }
        }
        //remember to add the last interval!!
        reslist.add(new int[]{min, max});
        return reslist.toArray(new int[0][0]);
    }
}
```

## 75. Sort Colors

**solution**
use double pointers

```java
class Solution {
    public void sortColors(int[] nums) {
        //double pointers
        //left points to the next position of the right bound of '0'
        //right points to the previous position of the left bound of '2'
        int left = 0, right = nums.length - 1;
        for(int i = 0; i <= right; i++){
            if (nums[i] == 0)
                swap(nums, left++, i);
            else if (nums[i] == 2)
                swap(nums, i--, right--);
        }
    }
    public void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```
## 121. Best Time to Buy and Sell Stock


**solution**

```java
class Solution {
    public int maxProfit(int[] prices) {
        int res = 0;
        int p = 0;
        for(int i = 0; i < prices.length; i++){
            res = Math.max(res, prices[i] - prices[p]);
            if (prices[i] < prices[p])
                p = i;
        }
        return res;
    }
}

----------or---------------------

class Solution {
    public int maxProfit(int[] prices) {
        int res = 0;
        int min = Integer.MAX_VALUE;
        for(int i = 0; i < prices.length; i++){
            res = Math.max(res, prices[i] - min);
            if (prices[i] < min)
                min = prices[i];
        }
        return res;
    }
}
```

## 128. Longest Consecutive Sequence
**solution**
Approach:  HashSet

- Time Complexity : O(n)
- Space Complexity: O(n)

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for(int num : nums)
            set.add(num);
        int res = 0;
        for(int num : nums){
            if (!set.contains(num - 1)){
                int len = 0;
                int cur = num;
                while(set.contains(cur)){
                    len++;
                    cur++;
                }
                res = Math.max(res, len);    
            }
            
        }
        return res;
    }
}
```

## 152. Maximum Product Subarray
**solution**
Approach: DP
max[i] = max(nums[i], nums[i] * max[i - 1], nums[i] * min[i - 1])
min[i] = min(nums[i], nums[i] * max[i - 1], nums[i] * min[i - 1])
max[i] represents the max product of the array which ends with nums[i]
same for min[i]
res = max(max[i]) for i >= 0 and i < n

```java
class Solution {
    public int maxProduct(int[] nums) {
        int res = Integer.MIN_VALUE;
        int maxPro = 1, minPro = 1;
        for(int num : nums){
            int maxProSave = maxPro;
            maxPro = Math.max(num, 
                              Math.max(num * maxPro, num * minPro));
            minPro = Math.min(num, 
                              Math.min(num * maxProSave, num * minPro));
            res = Math.max(res, maxPro);
        }
        return res;
    }
}
```

## 169. Majority Element
**solution**

Approach: Boyer-Moore Voting Algorithm

- Time complexity: O(n)
- Space complexity: O(1)

```java
class Solution {
    public int majorityElement(int[] nums) {
        //any initial value is ok
        //because it will be always set to the first element
        //after the first loop
        int candidate = 0;
        int count = 0;
        for(int num : nums){
            if (count == 0)
                candidate = num;
            count += (candidate == num) ? 1 : -1;
        }
        return candidate;
    }
}
```

## 238. Product of Array Except Self
**solution**
Approach: O(1) space approach

- Time complexity: O(n)
- Space complexity: O(1) (The input and output array does not count as extra space for the purpose of space complexity analysis.)

```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];
        res[0] = 1;
        for(int i = 1; i < nums.length; i++)
            res[i] = res[i - 1] * nums[i - 1];
        int R = 1;
        for(int i = nums.length - 1; i >= 0; i--){
            res[i] = R * res[i];
            R = R * nums[i];
        }
        return res;
    }
}
```

## 239. Sliding Window Maximum
**solution**
Approach: Deque

- Time complexity: O(n)
- Space complexity: O(n)

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int[] res = new int[nums.length - k + 1];
        Deque<Integer> deque = new ArrayDeque<>();
        for(int i = 0; i < nums.length; i++){
            //remove the item which is out of range of the current k-range window
            while(!deque.isEmpty() && deque.peek() < i - k + 1)
                deque.poll();
            //remove smaller numbers in k range as they are useless
            while(!deque.isEmpty() &&
                  nums[i] > nums[deque.peekLast()])
                deque.pollLast();
            deque.offer(i);
            //the head of the deque is the max value of current window
            if (i >= k - 1)
                res[i - k + 1] = nums[deque.peek()];
            
        }
        return res;
    }
}
```

## 283. Move Zeroes
**solution**


- Time Complexity: O(n)
- Space Complexity: O(1)

```java
class Solution {
    public void moveZeroes(int[] nums) {
        /*
         note that we can't use left bount of zero and decrease it,
         or we can't maintain the relative order of the non-zero elements
         right bound of non-zero.
         example: [0,1,0,3,12]
         both ptr = len - 1, i--(i from len - 1) and ptr = len - 1, i++(i from 0)
         are wrong
        */
        int ptr = 0;
        for(int i = 0; i < nums.length; i++){
            if (nums[i] != 0)
                swap(nums, i, ptr++);
        }
    }
    private void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```

## 287. Find the Duplicate Number

Note:
You must not modify the array (assume the array is read only).
You must use only constant, O(1) extra space.
Your runtime complexity should be less than O(n2).
There is only one duplicate number in the array, but it could be repeated more than once.

**solution**

Approach: fast and slow pointer

- Time Complexity: O(n)
- Space Complexity: O(1)

```java
class Solution {
    public int findDuplicate(int[] nums) {
        int slow = nums[0];
        int fast = nums[0];
        do{
            slow = nums[slow];
            fast = nums[nums[fast]];
        }while(slow != fast);
        fast = nums[0];
        while(slow != fast){
            slow = nums[slow];
            fast = nums[fast];
        }
        return fast;
    }
}
```

## 295. Find Median from Data Stream
**solution**
Approach: Two heaps

- Time Complexity: O(log n)
- Space Complexity: O(n)

```java
class MedianFinder {

    /** initialize your data structure here. */
    private PriorityQueue<Integer> max;
    private PriorityQueue<Integer> min;
    public MedianFinder() {
        max = new PriorityQueue<>((i1, i2) -> (i2 - i1));
        min = new PriorityQueue<>();
    }
    
    public void addNum(int num) {
        max.offer(num);
        min.offer(max.poll());
        if (max.size() < min.size())
            max.offer(min.poll());
    }
    
    public double findMedian() {
        if (max.size() == min.size())
            return (max.peek() + min.peek()) / 2.0;
        else
            return max.peek();
    }
}

```

## 309. Best Time to Buy and Sell Stock with Cooldown
**solution**
Approach DP

state   --action-->  next state(state: [hold, sold, rest]. action: [buy, sell, rest])
hold   --sell-->        sold
hold   --rest-->       hold
sold   --rest-->        rest
rest    --rest-->        rest
rest    --buy-->        hold

sold[i] = hold[i - 1] + price[i]; (max profit when i th day ends and in sold state, 0 <=i < n) 
hold[i] = max(rest[i - 1] - price[i], hold[i - 1])
rest[i] = max(sold[i - 1], rest[i - 1])

```java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[] hold = new int[n + 1];
        int[] sold = new int[n + 1];
        int[] rest = new int[n + 1];
        //initial state we don't have stock, this state is impossible, set as INT_MIN
        hold[0] = Integer.MIN_VALUE;
        for(int i = 0; i < n; i++){
            hold[i + 1] = Math.max(hold[i], rest[i] - prices[i]);
            sold[i + 1] = hold[i] + prices[i];
            rest[i + 1] = Math.max(rest[i], sold[i]);
        }
        return Math.max(hold[n], Math.max(rest[n], sold[n]));
    }
}
//-----------------------or-----------------------------------
class Solution {
    public int maxProfit(int[] prices) {
        int sold = 0, rest = 0, hold = Integer.MIN_VALUE;
        for(int price : prices){
            int soldPre = sold;
            sold = hold + price;
            hold = Math.max(rest - price, hold);
            rest = Math.max(soldPre, rest);
        }
        return Math.max(sold, Math.max(hold, rest));
    }
}
```

## 406. Queue Reconstruction by Height
**solution**

Approach: 

1. Pick out tallest group of people and sort them in a subarray (S). Since there's no other groups of people taller than them, therefore each guy's index will be just as same as his k value.
2. For 2nd tallest group (and the rest), insert each one of them into (S) by k value. So on and so forth.
E.g.
input: [[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
subarray after step 1: [[7,0], [7,1]]
subarray after step 2: [[7,0], [6,1], [7,1]]

- Time Complexity: O(n^2)
- Space Complexity: O(n)

```java
class Solution {
    public int[][] reconstructQueue(int[][] people) {
        List<int[]> res = new ArrayList<>();
        Arrays.sort(people, (o1, o2) -> 
                    o1[0] != o2[0] ? o2[0] - o1[0] : o1[1] - o2[1]);
        for(int[] p : people)
            res.add(p[1], p);
        return res.toArray(new int[0][0]);
    }
}
```

## 448. Find All Numbers Disappeared in an Array
**soultion**
Approach1: Similar to 41. First Missing Positive

- Time Complexity: O(n)
(each cell will be touched at most 2 times after that for loop is finished execution.)
- Space Complexity: O(1) (assume the returned list does not count as extra space.)

```java
class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for(int i = 0; i < nums.length; i++){
            while(nums[nums[i] - 1] != nums[i])
                swap(nums, i, nums[i] - 1);
        }
        for(int i = 0; i < nums.length; i++){
            if (nums[i] != i + 1)
                res.add(i + 1);
        }
        return res;
    }
    private void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```

Approach 2: set value of pos i to negative to mark the existence of the number i + 1

- Time Complexity: O(n)
- Space Complexity: O(1) (assume the returned list does not count as extra space.)

```java
class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for(int i = 0; i < nums.length; i++){
            int val = Math.abs(nums[i]);
            nums[val - 1] = - Math.abs(nums[val - 1]);
            
        }
        for(int i = 0; i < nums.length; i++)
            if (nums[i] > 0)
                res.add(i + 1);
        return res;
    }
}
```

## 560. Subarray Sum Equals K
**solution**
Approach : Using Hashmap

- Time Complexity: O(n)
- Space Complexity: O(n)

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> preSum = new HashMap<>();
        preSum.put(0, 1);
        int sum = 0, res = 0;
        for(int num : nums){
            sum += num;
            if (preSum.containsKey(sum - k))
                res += preSum.get(sum - k);
            preSum.put(sum, preSum.getOrDefault(sum , 0) + 1);
        }
        return res;
    }
}
```
