# What a person who does not know machine learning asks of this dashboard

Walked every tab of the live dashboard as a reader with no ML background, writing down every
question as it arose. Nothing here is a suggestion; it is the list of things the screen makes
a person wonder. Where a question turns out to point at a real defect rather than at missing
knowledge, it is marked **[DEFECT]** — those are worth more than the rest, because a person
who is confused can ask, while a screen that is wrong is believed.

Grouped by where the question appears, in the order a person meets them.

---

## The header

1. What is `final-s1-20260817-112115`? Is that a name, a date, or an ID?
2. What is an "iteration", and why are there 11.4 thousand of them? Is that a lot?
3. `48.5k шаг/с` — steps per second of what? Is the robot taking 48,500 steps a second?
4. **[DEFECT]** The remaining time keeps changing: 3h 37m, then 3h 23m, then 3h 38m, then
   3h 24m, all within a minute. Is it 3h 23m or 3h 38m? Which do I believe?
5. **[DEFECT]** Half the words are Russian (Консоль, Журнал, Сравнение, ИДЁТ ОБУЧЕНИЕ,
   ПРОГОН) and half are English (Episode return, Course success rate, Throughput). Whichever
   language I read, I hit the other one constantly. Is one of them for me?
6. There is a list of 20+ runs in the sidebar. Which one matters? Are the others failures,
   or history, or still going?

## "Кривые" — the charts

7. What does the x-axis mean? It reads `245.8k`, `350.25M`, `700.25M`, `1.22B`. A billion
   of what? And why does it start at 245.8k rather than at zero?
8. **Episode return** is described as "the headline learning curve" and it climbs to 6,000.
   Six thousand what? Is 6,000 good? What number would mean "finished"?
9. **[DEFECT]** **Course success rate** is a flat line at 0.00 across the entire run, and its
   description says "the definition of done". So the definition of done is zero, forever?
   Is the training failing completely? *(In fact this metric belongs to a different task and
   does not apply to walking at all — but nothing on screen says so, and it is placed second,
   directly beside the headline chart.)*
10. **Episode length** — "steps before falling or hitting the time limit". It sits at 2.3k.
    Is that falling or is that the time limit? The chart cannot tell me which, and those are
    opposite outcomes.
11. **Throughput** — "a sustained drop usually means thermal throttling". Should I be doing
    something if it drops? Is anyone watching this, or am I?
12. **[DEFECT]** **Reward terms** shows 21 coloured lines with a 21-item legend, on one
    chart, most of them overlapping near zero. The description says the point is "finding the
    one term that quietly dominates". I cannot tell any line from any other. Which one is
    dominating?
13. **Policy update health** — what is "KL"? What is a "clip fraction"? The description says
    KL "should track the 0.01 target" and the chart's axis goes to 0.30. Is 0.30 bad?
14. **Learning rate and exploration** — "action std falling to near zero means exploration has
    collapsed". What is exploration, why would it collapse, and what happens if it does?
15. **Losses** — "value loss should fall and stabilise". Mine goes up to 12,000. Is that
    falling? Is that stabilised?
16. "Entropy falling fast is an early warning of premature convergence." Warning of what,
    concretely? What would I do about it?
17. **Time breakdown** — physics, PPO update, inference. Why do I need to know this? Is it
    for me or for the person who wrote it?

## "Походка" — the gait score

18. **[DEFECT]** The big number says **17% HUMAN-LIKE**, with **▼ 8 pts** beside it and
    **best 40%**. So it was 40% and now it is 17%? It has more than halved. Is the robot
    getting *worse*? Nothing on this screen says whether that is expected.
19. **[DEFECT, and the sharpest one]** The first tab's headline number (return) rises all
    run. This tab's headline number (human-like) falls. **The two most prominent numbers on
    the dashboard point in opposite directions and neither mentions the other.** Which one
    is the truth?
20. "Scored against measured human walking, not against reward." What is the difference, and
    why are there two different scores at all?
21. **Torso upright 0.77 vs 0.95-1.00** — 0.77 of what? Percent? A ratio? 0.77 out of 1?
22. **Stance width 31.41cm vs 10.00-15.00** — is 31 cm the distance between his feet? That
    sounds like standing normally. Why is 12 cm the human number? *(Worth checking: a person
    measuring their own feet gets ~10 cm and would immediately doubt the 31.)*
23. **[DEFECT]** **Balance 0%** is a whole category reading zero, and its only number is
    "Left/right evenness 0.73, human 0.90 to 1.00". How does 0.73 against a target of 0.90
    become **zero percent**? That looks like a broken calculation, not a bad robot.
24. Same for **Reliability 11%** while "staying upright 0.08 vs human 0.00-0.05" — 0.08
    against 0.05 is close. Why is the category at 11%?
25. **Vertical bounce 2.58cm vs human 4.00-5.00cm** — he bounces *less* than a human and that
    is scored as a fault? Should I want him to bounce more?
26. "Jaggedness is expected: RL oscillates." Expected by whom? How jagged is too jagged?
27. Six lines on the score chart, all crossing. Which one is the one to watch?

## "Обучение" — what changed and why

28. This tab explains Collect / Judge / Nudge / Check. It is the clearest thing on the
    dashboard. **Why is it not the first tab?** By the time I reach it I have already been
    shown eleven charts I could not read.
29. "The critic predicted how good each moment would be." There is a critic? Is that a second
    robot? Who is it?
30. "Advantage: positive means better than expected." Better than *what* expected — better
    than last time, or better than the prediction?
31. **Which weights changed?** — `actor.4`, `actor.0`, `critic.2`. What are these? Should the
    output layer moving most be reassuring or alarming? The text says it is normal, but it
    also says it moves 340%, and 340% of anything sounds broken.
32. "So the two lines below should mirror each other." Do they? I cannot tell — they are on
    the same axis with values 0 to 160 and different meanings.

## "Команды" — does it do what it is told

33. **[DEFECT, and the most useful screen on the dashboard]** This table is immediately
    readable: *Forward — asked 0.88, delivered 0.76, 86% follows it.* **Why is this not the
    front page?** It is the only screen that answers "is it working" in a sentence.
34. **Turning: 16% — largely ignores it. Practised 65%.** So the thing it practises most is
    the thing it obeys least? Is that a problem? Nothing on screen says.
35. "Delivered is negative when it moves the opposite way to its command." It sometimes walks
    *backwards* when told to go forwards? How often?
36. `0.29 rad/s` — what is a radian per second? Is 0.29 fast?
37. "8% of attempts end in a fall" and "19.1s upright per attempt (limit 20s)". If it stays
    up 19.1 of 20 seconds, how does it fall 8% of the time? Are those the same attempts?

## "Сравнение" — the comparison table

38. **[DEFECT]** Every row reads **0.00%**, **0.00%**, and **0** in the last three columns.
    Twenty-four runs, all zero. Did everything fail?
39. **[DEFECT]** The two runs in the sidebar that are actually running right now
    (`final-s0`, `final-s1`) are **not in this table at all**. The comparison screen compares
    only old runs of a different task, and nothing says so. A person concludes the current
    work is failing, or missing.
40. Four of the columns are Russian words I cannot map to anything: ХОЛДЫ, СТОЙКА, СТРОГИЙ
    ПРЕДИКАТ, ЭКЗАМЕН. What is an exam? Who is examining?
41. The footnote says returns are not comparable between runs because the reward system
    changed (E31→E42). What is E31? Where do I read it?

## Across the whole thing

42. **What is this robot supposed to end up doing?** No screen states the goal. I can see
    dozens of measurements and not one sentence saying what success looks like.
43. **Is it going well right now?** There is no single answer anywhere. Return says yes,
    human-likeness says it got worse, comparison says everything is zero.
44. **Should I do anything?** Several descriptions hint at trouble ("a sustained drop usually
    means", "an early warning of") without saying who acts or when.
45. **How long until it is finished — not this run, the whole thing?** The header counts down
    3 hours. Is the project 3 hours from done?
46. What is the difference between "iterations", "steps", "episodes" and "evaluations"? All
    four are on screen and all four count something different.
47. Twenty-two videos. Which is the newest? Which shows the current state? Do I watch all 22?

---

## The five that are worth fixing first

Ordered by how badly they mislead, not by effort.

1. **The two headline numbers contradict each other** (Q19). Return rises, human-likeness
   falls, neither mentions the other, and both are presented as the top-level answer.
2. **"Course success rate" is a flat zero on a chart labelled "the definition of done"**
   (Q9) — it is a different task's metric, shown second, with nothing marking it as
   inapplicable.
3. **The comparison tab excludes the runs that are running and shows 24 rows of zeros**
   (Q38, Q39).
4. **Balance reads 0% for a value of 0.73 against a 0.90 target** (Q23). Either the scale is
   wrong or it needs explaining; as shown it reads as a broken number.
5. **The two clearest screens — "Обучение" and "Команды" — are fifth and eighth** (Q28,
   Q33). The order teaches the reader that the dashboard is not for them before they reach
   the parts that are.

And one that is not a defect but is the reason for most of the list: **nothing on any screen
says what the robot is supposed to be able to do**. Every number is a distance from an
unstated destination.
