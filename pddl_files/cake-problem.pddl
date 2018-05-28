; The "have cake and eat it too" problem.
; Good case to test failure since it can't be solved.

(define (problem HaveCakeAndEatItToo)
   (:domain Cake)

   (:init (have Cake) )

   (:goal (and (have Cake) (eaten Cake)))
)
