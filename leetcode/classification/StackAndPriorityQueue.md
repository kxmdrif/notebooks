# StackAndPriorityQueue

## 84. Largest Rectangle in Histogram
**solution**
Approach 1
use extra array

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
         if (heights == null || heights.length == 0) 
             return 0;
        
        //first index of the bar the left less than the current bar
        int[] lessFromLeft = new int[heights.length];
        //similarily
        int[] lessFromRight = new int[heights.length];
        lessFromLeft[0] = -1;
        lessFromRight[heights.length - 1] = heights.length;
        for(int i = 1; i < heights.length; i++){
            int p = i - 1;
            while(p >= 0 && heights[p] >= heights[i])
                p = lessFromLeft[p];
            lessFromLeft[i] = p;
        }
        for(int i = heights.length - 2; i >= 0; i--){
            int p = i + 1;
            while(p < heights.length && heights[p] >= heights[i])
                p = lessFromRight[p];
            lessFromRight[i] = p;
        }
        int maxArea = 0;
        for(int i = 0; i < heights.length; i++)
            maxArea = Math.max(maxArea, heights[i] * (lessFromRight[i] - lessFromLeft[i] - 1));
        return maxArea;
    }
}
```

Approach 2
use increasing stack

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int res = 0;
        // i <= heights.length !!!
        for(int i = 0; i <= heights.length; i++){
            int h = (i == heights.length ? 0 : heights[i]);
            if (stack.isEmpty() || h >= heights[stack.peek()]){
                stack.push(i);
            }else{
                int tp = stack.pop();
                res = Math.max(res, 
                    heights[tp] * (stack.isEmpty() ? i : i - stack.peek() - 1));
                i--;
            }
        }
        return res;
    }
}
```

## 155. Min Stack

```java
class MinStack {

    /** initialize your data structure here. */
    class Node{
        int val;
        int min;
        Node(int val, int min){
            this.val = val;
            this.min = min;
        }
    }
    private Stack<Node> stack;
    public MinStack() {
        stack = new Stack<>();
    }
    
    public void push(int x) {
        
        if (stack.isEmpty())
            stack.push(new Node(x, x));
        else
            stack.push(new Node(x, Math.min(x, stack.peek().min)));
            
    }
    
    public void pop() {
        stack.pop();
    }
    
    public int top() {
        return stack.peek().val;
    }
    
    public int getMin() {
        return stack.peek().min;
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
```

## 215. Kth Largest Element in an Array
Approach 1: Sort

- Time complexity: O(nlogn)
- Space complexity: O(1)
```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        Arrays.sort(nums);
        return nums[nums.length - k];
    }
}
```

Approach 2: Heap Sort

- Time complexity: O(nlogk)
- Space complexity: O(k)
```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for(int num : nums){
            pq.offer(num);
            if (pq.size() > k)
                pq.poll();
        }
        return pq.peek();
    }
}
```


Approach 3: Quick select

- Time complexity: Best O(n), Worst O(n^2)
- Space complexity: O(1)

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        return quickSelect(nums, nums.length - k + 1, 0,
                           nums.length - 1);
    }
    
    //find kth smallest
    private int quickSelect(int[] nums, int k, int l, int r){
        int index = partation(nums, l, r);
        if (l + k - 1 == index)
            return nums[index];
        else if (l + k - 1 > index)
            return quickSelect(nums, k + l - index - 1, index + 1, r);
        else 
            return quickSelect(nums, k, l, index - 1);
    }
    
    private int partation(int[] nums, int l, int r){
        int pivot = l;
        int index = pivot + 1;
        for(int i = index; i <= r; i++){
            if (nums[i] < nums[pivot]){
                swap(nums, i, index);
                index++;
            }
        }
        swap(nums, l, index - 1);
        return index - 1;
    }
    
    private void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```

## 347. Top K Frequent Elements
**solution**

Approach 1 :  Heap

- Time Complexity: O(nlogk)
- Space Complexity: O(n + k) (store the hash map with not more N elements and a heap with k elements)

```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int num : nums)
            map.put(num, map.getOrDefault(num, 0) + 1);
        PriorityQueue<Integer> pq = new PriorityQueue<>((n1, n2) -> 
                                    (map.get(n1) - map.get(n2)));
        for(int n : map.keySet()){
            pq.offer(n);
            if (pq.size() > k) pq.poll();
        }
        int[] res = new int[k];
        for(int i = 0; i < k; i++)
            res[i] = pq.poll();
        return res;
        
    }
}
```

Approach 2 :  Quick Select

- Time Complexity: O(n)(average), O(n^2)(worst)
- Space Complexity: O(n) (store hash map and array of unique element)

```java
class Solution {
    private Map<Integer, Integer> count;
    public int[] topKFrequent(int[] nums, int k) {
        count = new HashMap<>();
        for(int num : nums)
        count.put(num, count.getOrDefault(num, 0) + 1);
        int[] unique = new int[count.keySet().size()];
        int i = 0;
        for(int num : count.keySet())
            unique[i++] = num;
        int n = unique.length;
        quickSelect(unique, 0, n - 1, n - k + 1);
        return Arrays.copyOfRange(unique, n - k, n);
        
        
    }
    
    //note: we we find the kth, the part before the kth is smaller than it
    //and the part after the kth is larger than it.
    private void quickSelect(int[] nums, int l, int r, int k){
        int idx = partition(nums, l, r);
        if (l + k - 1 == idx)
            return;
        else if (l + k - 1 > idx)
            quickSelect(nums, idx + 1, r, l + k - idx - 1);
        else
            quickSelect(nums, l, idx - 1, k);
    }
    private int partition(int[] nums, int l, int r){
        int pivot = l;
        int idx = l + 1;
        for(int i = l + 1; i <= r; i++){
            if (count.get(nums[i]) < count.get(nums[pivot]))
                swap(nums, i, idx++);
        }
        idx = idx - 1;
        swap(nums, pivot, idx);
        return idx;
    }
    private void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```

Approach 3 :  Bucket Select

- Time Complexity: O(n)
- Space Complexity: O(n)

```java
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> count = new HashMap<>();
        for(int num : nums)
            count.put(num, count.getOrDefault(num, 0) + 1);
        //new List<>[nums.length + 1] and new List<Integer>[nums.length + 1]
        //is wrong!!
        List<Integer>[] bucket = new List[nums.length + 1];
        for(int num : count.keySet()){
            int freq = count.get(num);
            if (bucket[freq] == null)
                bucket[freq] = new ArrayList<>();
            bucket[freq].add(num);
        }
        int[] res = new int[k];
        int p = 0;
        for(int i = nums.length; i >= 0; i--){
            if (bucket[i] == null) continue;
            for(int num : bucket[i]){
                res[p++] = num;
                if (p == k) return res;
            }
        }
        return res;
    }
    
}
```