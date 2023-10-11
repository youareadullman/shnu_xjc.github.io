## 485
给定一个二进制数组 nums ， 计算其中最大连续 1 的个数。
```
示例 1：
输入：nums = [1,1,0,1,1,1]
输出：3
```
```java
class Solution {
    public int findMaxConsecutiveOnes(int[] nums) {
        int maxCount = 0, count = 0;
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (nums[i] == 1) {
                count++;
            } else {
                maxCount = Math.max(maxCount, count);
                count = 0;
            }
        }
        maxCount = Math.max(maxCount, count);
        return maxCount;
    }
}
```
