# Backtracking

## 17. Letter Combinations of a Phone Number

**solution**

Approach 1: Backtracking

- Time Complexity: O(3^M * 4 ^ N) where N is the number of digits in the input that maps to 3 letters (e.g. 2, 3, 4, 5, 6, 8) and M is the number of digits in the input that maps to 4 letters (e.g. 7, 9), and N+M is the total number digits in the input.

- Space Complexity: O(3^M * 4 ^ N) since one has to keep 3^N * 4^M solutions.

```java
import java.util.*;
class Solution {
    private List<String> res = new ArrayList<>();
    private int len = 0;
    String[] map = new String[]{"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    
    public List<String> letterCombinations(String digits) {
        len = digits.length();
        if (len == 0) return res;
        rec(0, new char[len], digits.toCharArray());
        return res;
    }
    private void rec(int i, char[] single, char[] digits){
        if (i == len){
            res.add(new String(single));
            return;
        }
        for(char c : map[digits[i] - '2'].toCharArray()){
            single[i] = c;
            rec(i + 1, single, digits);
        }
    }
    
}
```

Approach 2 : use queue

```java
import java.util.*;
class Solution {
    public List<String> letterCombinations(String digits) {
        String[] map = new String[]{"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        LinkedList<String> res = new LinkedList<>();
        
        if (digits.isEmpty()) return res;
        res.add("");
        
        while(res.peek().length() != digits.length()){
            String s = res.poll();
            for(char c : 
                map[digits.charAt(s.length()) - '2'].toCharArray())
                
                res.offer(s + c);
            
        }
        return res;
    }
    
}
```

## 39. Combination Sum
**solution**
Approach: Backtracking

```java
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        backtrack(res, new ArrayList<>(), target, 0, candidates);
        return res;
    }
    /*
    1 use begin to avoid duplicates
    2 copy the combination list at last instead of every loop to
    decrease the cost of copying a list
    */
    private void backtrack(List<List<Integer>> res, List<Integer> comb, int target, int begin, int[] candidates){
        if (target < 0) return;
        if (target == 0){
            res.add(new ArrayList<>(comb));
            return;
        }
        for(int i = begin; i < candidates.length; i++){
            comb.add(candidates[i]);
            backtrack(res, comb, target - candidates[i], i, candidates);
            comb.remove(comb.size() - 1);
        }
    }
}
```

## 46. Permutations
**solution**
Approach: Backtracking

```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        int[] used = new int[nums.length];
        rec(res, new ArrayList<>(), used, nums);
        return res;
    }
    private void rec(List<List<Integer>> res,
                     List<Integer> curr, int[] used, int[] nums){
        if (curr.size() == nums.length){
            res.add(new ArrayList<>(curr));
            return;
        }
        for(int i = 0; i < nums.length; i++){
            if(used[i] == 0){
                used[i] = 1;
                curr.add(nums[i]);
                rec(res, curr, used, nums);
                curr.remove(curr.size() - 1);
                used[i] = 0;
            }
        }
    }
}
```

## 78. Subsets
**solution**
Approach 1: Lexicographic (Binary Sorted) Subsets

- Time complexity: O(n * 2^n)
- Space complexity: O(n * 2 ^ n)

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;
        
        for(int i = (int) Math.pow(2, n); i < (int) Math.pow(2, n + 1); i++){
            String bitmask = Integer.toBinaryString(i).substring(1);
            List<Integer> subset = new ArrayList<>();
            for(int j = 0; j < n; j++)
                if (bitmask.charAt(j) == '1')
                    subset.add(nums[j]);
            res.add(subset);
        }
        return res;
    }
}
```

Approach 2 : Backtracking

- Time complexity: O(n * 2^n)
- Space complexity: O(n * 2 ^ n)

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        
        List<List<Integer>> res = new ArrayList<>();
        backtrack(res, new ArrayList<>(), 0, nums);
        return res;
    }
    
    private void backtrack(List<List<Integer>> res,
                List<Integer> subset, int begin, int[] nums){
        res.add(new ArrayList<>(subset));
        for(int i = begin; i < nums.length; i++){
            subset.add(nums[i]);
            backtrack(res, subset, i + 1, nums);
            subset.remove(subset.size() - 1);
        }
    }
}
```