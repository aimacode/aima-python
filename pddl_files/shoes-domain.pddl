;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Putting on a pair of shoes
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain Shoes)
  (:requirements :strips)
  (:predicates (leftfoot ?x)
           (rightfoot ?x)
	       (on ?x ?y)
  )

  (:action Right_Shoe_On
	     :parameters ()
	     :precondition (and (on RightSock ?x) (rightfoot ?x) (not (on RightShoe ?x)))
	     :effect (and (on RightShoe ?x))
  )

  (:action Right_Sock_On
         :parameters ()
         :precondition (and (clear ?x) (rightfoot ?x))
         :effect (and (on RightSock ?x) (not (clear ?x)))
  )

  (:action Left_Shoe_On
         :parameters ()
         :precondition (and (on LeftSock ?x) (leftfoot ?x) (not (on LeftShoe ?x)))
         :effect (and (on LeftShoe ?x))
  )

  (:action Left_Sock_On
         :parameters ()
         :precondition (and (clear ?x) (leftfoot ?x))
         :effect (and (on LeftSock ?x) (not (clear ?x)))
  )

)

