#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Answer 1:

def merge_intervals(intervals):
    # Sort the intervals based on start times
    intervals.sort(key=lambda x: x[0])

    merged = []
    prev_start, prev_end = intervals[0]

    for start, end in intervals[1:]:
        # Check if the current interval overlaps with the previous merged interval
        if start <= prev_end:
            prev_end = max(prev_end, end)  # Merge intervals
        else:
            merged.append([prev_start, prev_end])
            prev_start, prev_end = start, end

    merged.append([prev_start, prev_end])  # Add the last merged interval

    return merged


# In[2]:


intervals = [[1,3],[2,6],[8,10],[15,18]]
result = merge_intervals(intervals)
print(result)


# In[3]:


intervals = [[1,4],[4,5]]
result = merge_intervals(intervals)
print(result) 


# In[4]:


#Answer2:

def sortColors(nums):
    left = 0
    mid = 0
    right = len(nums) - 1

    while mid <= right:
        if nums[mid] == 0:
            nums[left], nums[mid] = nums[mid], nums[left]
            left += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[right] = nums[right], nums[mid]
            right -= 1

    return nums


# In[5]:


nums = [2, 0, 2, 1, 1, 0]
result = sortColors(nums)
print(result) 

nums = [2, 0, 1]
result = sortColors(nums)
print(result) 


# In[22]:


#Answer3:
def firstBadVersion(n):
    left = 1
    right = n

    while left < right:
        mid = left + (right - left) // 2
        if firstBadVersion(mid):
            right = mid
        else:
            left = mid + 1

    return left


# In[23]:


n = 5
bad = 4
result = firstBadVersion(n)
print(result)  # Output: 4

n = 1
bad = 1
result = firstBadVersion(n)
print(result)  # Output: 1


# In[24]:


#Answer4:
def maximumGap(nums):
    n = len(nums)
    if n < 2:
        return 0

    # Step 2: Find min and max values
    min_val = min(nums)
    max_val = max(nums)

    # Step 3: Calculate bucket size
    bucket_size = max(1, (max_val - min_val) // (n - 1))

    # Step 4: Create buckets
    buckets = [[float('inf'), float('-inf')] for _ in range((max_val - min_val) // bucket_size + 1)]

    # Step 5: Place elements in buckets
    for num in nums:
        idx = (num - min_val) // bucket_size
        buckets[idx][0] = min(buckets[idx][0], num)
        buckets[idx][1] = max(buckets[idx][1], num)

    # Step 6: Calculate maximum gap
    prev_max = min_val
    max_gap = 0

    # Step 7: Iterate through buckets
    for bucket in buckets:
        if bucket[0] != float('inf'):
            max_gap = max(max_gap, bucket[0] - prev_max)
            prev_max = bucket[1]

    # Step 8: Return the maximum gap
    return max_gap


# In[25]:


nums = [3, 6, 9, 1]
result = maximumGap(nums)
print(result)  

nums = [10]
result = maximumGap(nums)
print(result) 


# In[26]:


#Answer5:
def containsDuplicate(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False


# In[27]:


nums = [1, 2, 3, 1]
result = containsDuplicate(nums)
print(result) 

nums = [1, 2, 3, 4]
result = containsDuplicate(nums)
print(result) 

nums = [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
result = containsDuplicate(nums)
print(result)  


# In[29]:


#Answer6:
def findMinArrowShots(points):
    points.sort(key=lambda x: x[1])  # Sort balloons based on end points
    arrows = 1
    end = points[0][1]
    for i in range(1, len(points)):
        if points[i][0] > end:
            arrows += 1
            end = points[i][1]
    return arrows


# In[30]:


points = [[10, 16], [2, 8], [1, 6], [7, 12]]
result = findMinArrowShots(points)
print(result)  
points = [[1, 2], [3, 4], [5, 6], [7, 8]]
result = findMinArrowShots(points)
print(result)  
points = [[1, 2], [2, 3], [3, 4], [4, 5]]
result = findMinArrowShots(points)
print(result)  


# In[31]:


#Answer7:

def lengthOfLIS(nums):
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


# In[32]:


nums = [10, 9, 2, 5, 3, 7, 101, 18]
result = lengthOfLIS(nums)
print(result) 

nums = [0, 1, 0, 3, 2, 3]
result = lengthOfLIS(nums)
print(result)

nums = [7, 7, 7, 7, 7, 7, 7]
result = lengthOfLIS(nums)
print(result)


# In[33]:


#Answer8:

def find132pattern(nums):
    stack = []
    numk = float('-inf')
    n = len(nums)

    for i in range(n - 1, -1, -1):
        if nums[i] < numk:
            return True
        while stack and nums[i] > stack[-1]:
            numk = stack.pop()
        stack.append(nums[i])

    return False


# In[34]:


nums = [1, 2, 3, 4]
result = find132pattern(nums)
print(result) 

nums = [3, 1, 4, 2]
result = find132pattern(nums)
print(result) 

nums = [-1, 3, 2, 0]
result = find132pattern(nums)
print(result) 


# In[ ]:




