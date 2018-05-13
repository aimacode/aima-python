;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Changing a spare tire on a car
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain SpareTire)
  (:requirements :strips)
  (:predicates (At Spare Trunk)
               (At Spare Ground)
               (At Flat Axle)
               (At Flat Ground)
               (At Spare Axle))

  (:action remove
	     :parameters (Spare Trunk)
	     :precondition (At Spare Trunk)
	     :effect (and (At Spare Ground) (not (At Spare Trunk))))

  (:action remove
	     :parameters (Flat Axle)
	     :precondition (At Flat Axle)
	     :effect (and (At Flat Ground) (not (At Flat Axle))))

  (:action put_on
	     :parameters (Spare Axle)
	     :precondition (and (At Spare Ground) (not (At Flat Axle)))
	     :effect (and (At Spare Axle) (not (At Spare Ground))))

  (:action leave_overnight
	     :effect
	     (and (not (At Spare Ground))
	          (not (At Spare Axle))
	          (not (At Spare Trunk))
	          (not (At Flat Ground))
	          (not (At Flat Axle))
	     )
  )
)
