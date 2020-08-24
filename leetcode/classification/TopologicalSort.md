# TopologicalSort

## 207. Course Schedule
**solution**
Approach: Topological Sort

```java
import java.util.*;
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        Map<Integer, List<Integer>> edge = new HashMap<>();
        int[] inDegree = new int[numCourses];
        for(int i = 0; i < numCourses; i++)
                edge.put(i, new ArrayList<>());
                
        for(int[] pre : prerequisites){
            edge.get(pre[1]).add(pre[0])
            inDegree[pre[0]]++;
        }
        
        return topologySort(edge, inDegree);
    }
    
    private boolean topologySort(Map<Integer, List<Integer>> edge, int[] inDegree){
        int n = inDegree.length;
        Queue<Integer> queue = new LinkedList<>();
        int count = 0;
        for(int i = 0; i < n; i++)
            if(inDegree[i] == 0)
                queue.offer(i);
        while(!queue.isEmpty()){
            int node = queue.poll();
            count++;
            for(Integer adj : edge.get(node)){
                inDegree[adj]--;
                if(inDegree[adj] == 0)
                    queue.add(adj);
            }
        }
        return count == n;
    }
}

//------------------------------or-------------------------------------

class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        //use array instead of map. Attention:
        //List<Integer>[] adjs = new ArrayList<Integer>[numCourses];
        //or = new ArrayList<>[numCourses] is wrong(generic array creation).
        List<Integer>[] adjs = new ArrayList[numCourses];
        int[] inDegree = new int[numCourses];
        for(int i = 0; i < numCourses; i++)
            adjs[i] = new ArrayList<>();
        for(int[] prereq : prerequisites){
            inDegree[prereq[0]]++;
            adjs[prereq[1]].add(prereq[0]);
        }
        return topologicalSort(adjs, inDegree);
            
    }
    
    private boolean topologicalSort(List<Integer>[] adjs,
                                    int[] inDegree){
        int n = inDegree.length;
        Queue<Integer> queue = new LinkedList<>();
        for(int i = 0; i < n; i++)
            if (inDegree[i] == 0)
                queue.offer(i);
        int count = 0;
        while(!queue.isEmpty()){
            int node = queue.poll();
            count++;
            for(int adj : adjs[node]){
                inDegree[adj]--;
                if (inDegree[adj] == 0)
                    queue.offer(adj);
            }
        }
        return count == n;
    }
}
```