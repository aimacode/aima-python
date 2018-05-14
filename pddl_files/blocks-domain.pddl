;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Building block towers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain BlocksWorld)
  (:requirements :strips)
  (:predicates (on ?x ?y)
	       (clear ?x)
	       (block ?x)
  )

  (:action Move
	     :parameters (?b ?x ?y)
	     :precondition (and (on ?b ?x) (clear ?b) (clear ?y) (block ?b))
	     :effect (and (on ?b ?y) (clear ?x) (not (on ?b ?x)) (not (clear ?y)))
  )

  (:action Move_To_Table
         :parameters (?b ?x)
         :precondition (and (on ?b ?x) (clear ?b) (block ?b))
         :effect (and (on ?b Table) (clear ?x) (not (on ?b ?x)))
   )
)

