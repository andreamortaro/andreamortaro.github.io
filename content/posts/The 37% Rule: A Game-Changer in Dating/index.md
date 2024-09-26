---
title: "The 37% Rule: A Game-Changer in Dating"
date: 2023-10-16
tags: ["math", "TIL"]
summary: "How many frogs must you kiss to get a prince or a princess?"
draft: false
---

{{< katex >}}

## Introduction

Finding "the One" isn't a straightforward task. To increase your chances, you might consider a simple mathematical rule: the *37% Rule*. This rule gives you a sense of how long you should keep looking and when it might be a good time to stop and settle down.

Balancing when to settle down is crucial. Settling down too early might mean missing better matches later, while waiting too long risks missing out on great partners.
Therefore, the central question is: **how many options do we need to consider before to make a choice?** 
This question comes up in many real-life decision-making processes, from renting an apartment to the search for a parking space.
Mathematicians refer to these as "Optimal Stopping problems", with the most well-known in literature being the "Secretary problem", where the aim is to select the best candidate for a job position.

The "Secretary problem" reflects past-century gender biases, while nowadays dating is a prevalent setting for Optimal Stopping problems. In this context we wonder how many frogs do you need to kiss to enhance your chances of finding a prince or princess.

## Strategy

In the dating scenario we consider simple rules:
* When you reject someone they are gone forever. You can not go back and change your mind.
* When you choose someone your dating process is over, and you can not see next potential partners in future dates.

The strategy proposed by the 37% rule involves a two-phase approach:
1. ***Exploring phase***: Explore the initial 37% of the applicants and gather data, without make commitments. This serves as a calibration period, allowing you to form an idea of the market and develop a realistic expectation for a potential life partner.

2. ***Leap phase***: Commit to someone better than all those you've encountered thus far in the process.

Roughly speaking, you should reject everyone in the the first third of your decision-making process. Then, after the exploration, selecting the next great option you encounter is optimal. It doesn't guarantee that it's the best one, but it's optimal because you have the highest chance that it's.

In fact, employing this strategy yields surprisingly a probability of 37% (more precisely, the probability is \\(\dfrac{1}{e} \approx 0.369\\) ) for selecting the best option from the available pool. <cite>[^proof]</cite> Due to its two-phased nature, this rule is often referred to as the **Look-Then-Leap Rule** <cite>[^book]</cite>.

You can appreciate the 37% rule when you compare it with a random strategy.
When you choose over \\(n\\) potential partners and you settle down with a partner at random, then you’d only have a \\(\dfrac{1}{n}\\) chance of finding your true love. For \\(n \geq 3\\), it becomes evident that this random approach is less advantageous when compared to the look-then-leap strategy.

Take a look at the Kepler experience, it's an interesting application of this rule. <cite>[^wife]</cite> 

## Alternative setting

Since you don't know how many people you will encounter in your lifetime, you can set a time frame for your search, based on how long you expect your dating life to be. And also in this setting the 37% rule works.
In fact, the 37% rule can be applied to either *the number of candidates* or *the time over which one is searching*.

For example, consider to start dating when you are 18 years old and you would like to settle down at 35 years old. You should reject everyone until just after your 24th birthday, which corresponds to the exploring phase. Then you choose the next person you date, who is better than everyone you have met before. 

There are several variations of this problem, discussed in the book "Algorithms to Live By: The Computer Science of Human Decisions" by Brian Christian and Tom Griffiths.<cite>[^book]</cite>


## Conclusion

<!-- I’m a mathematician and therefore biased, but this result literally blows my mind. -->

Life is undeniably complex and cannot be confined to a purely mathematical analysis. This rule does not consider the nuances of feelings, "gut instincts", and instant chemistry, as it only focuses on maximizing probabilities. It's essential to acknowledge that matters of the heart often defy logic, and this heuristic rule, while helpful, doesn't guarantee success.

<!-- Nevertheless life is too messy to be trapped into a cold mathematical analysis. This rule doesn’t account for feelings, “gut instincts” and instant chemistry. -->
<!-- It is true that matters of the heart aren't logical, and this heuristic rule only maximizes probabilities. It doesn't guarantee success. -->

Moreover, this strategy isn't limited to selecting a life partner; it's applicable to numerous scenarios where people seek something and want to determine the optimal moment to conclude their search. The essence of sound decision-making lies in striking a balance between exploration and decisiveness, and that's precisely what the 37 percent rule aids in achieving.

<!-- Beyond choosing a partner, this strategy also applies to a host of other situations where people are searching for something and want to know the best time to stop looking. -->
<!-- The trick to great decision making is balancing exploration and decisiveness, and that's exactly what the 37 percent rule helps you do. -->

## Sources
* [algorithmstoliveby.com](https://algorithmstoliveby.com/)
* [bigthink.com](https://bigthink.com/neuropsych/the-37-percent-rule/)
* [washingtonpost.com](https://www.washingtonpost.com/news/wonk/wp/2016/02/16/when-to-stop-dating-and-settle-down-according-to-math/)
* [ted.com](https://ideas.ted.com/when-should-you-settle-down/)
* [cantorsparadise.com](https://www.cantorsparadise.com/math-based-decision-making-the-secretary-problem-a30e301d8489)
* [sneaky-potato.github.io](https://sneaky-potato.github.io/til/37percent/)


[bt]: https://bigthink.com/neuropsych/the-37-percent-rule/
[^book]: For more details take a look at this book: https://algorithmstoliveby.com.
[^wife]: You can find how Kepler chose his wife [here](https://www.npr.org/sections/krulwich/2014/05/15/312537965/how-to-marry-the-right-girl-a-mathematical-solution).
[^proof]: For more details, you can find a proof [here](https://plus.maths.org/content/kissing-frog-mathematicians-guide-mating-0).

---

Photo by <a href="https://unsplash.com/@shairad?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Shaira Dela Peña</a> on <a href="https://unsplash.com/photos/pink-love-neon-signage-twoEJNpgdbI?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>.
