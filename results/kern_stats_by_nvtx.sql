-- 按 NVTX 区间统计 kernel（用法：把 <report>.sqlite 换成目标报告）
-- 原理：NVTX_EVENTS 有各标注段的 [start,end)，kernel 表 CUPTI_ACTIVITY_KIND_KERNEL 有每个 kernel 的 [start,end)，
--       join 时间窗即可"只统计区间内的内核"（等价于 --nvtx-capture 过滤的效果）

-- 1) 先看有哪些 NVTX 段（名字+时长）
SELECT (end-start)/1e6 AS dur_ms, text FROM NVTX_EVENTS ORDER BY start;

-- 2) 统计某段（'measure'）内按 kernel 分组的时间与次数
SELECT COUNT(*) AS inst,
       SUM(k.end-k.start)/1e9 AS total_s,
       AVG(k.end-k.start)/1e6 AS avg_ms,
       k.demangledName
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN NVTX_EVENTS n ON n.text LIKE '%measure%' AND k.start >= n.start AND k.end <= n.end
GROUP BY k.demangledName
ORDER BY total_s DESC LIMIT 15;

-- 3) 该段总 GPU kernel 时间（秒）
SELECT SUM(k.end-k.start)/1e9
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN NVTX_EVENTS n ON n.text LIKE '%measure%' AND k.start >= n.start AND k.end <= n.end;
