;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Putting on a pair of shoes
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain Shoes)
  (:requirements :strips)
  (:predicates (leftfoot ?x)
           (rightfoot ?x)
	       (on ?x ?y)
  )

  (:action RightShoe
	     :parameters ()
	     :precondition (and (on RightSock ?x) (rightfoot ?x) (not (on RightShoe ?x)))
	     :effect (and (on RightShoe ?x))
  )

  (:action RightSock
         :parameters ()
         :precondition (and (clear ?x) (rightfoot ?x))
         :effect (and (on RightSock ?x) (not (clear ?x)))
  )

  (:action LeftShoe
         :parameters ()
         :precondition (and (on LeftSock ?x) (leftfoot ?x) (not (on LeftShoe ?x)))
         :effect (and (on LeftShoe ?x))
  )

  (:action LeftSock
         :parameters ()
         :precondition (and (clear ?x) (leftfoot ?x))
         :effect (and (on LeftSock ?x) (not (clear ?x)))
  )

)

