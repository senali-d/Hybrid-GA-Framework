import numpy as np


class Lecturer:
    def __init__(self, id, name):
        self.id = id
        self.name = name


class StudentGroup:
    def __init__(self, id, name, size):
        self.id = id
        self.name = name
        self.size = size


class Room:
    def __init__(self, id, name, capacity):
        self.id = id
        self.name = name
        self.capacity = capacity


class Module:
    def __init__(self, id, name, lecturer, student_group):
        self.id = id
        self.name = name
        self.lecturer = lecturer
        self.student_group = student_group


class TimetablingProblem:
    """This class encapsulates the University Timetabling Problem (UTP)"""

    def convert_to_timetable_format(self, chromosome):
        """Decode integer chromosome into (module_idx, room_idx, timeslot_idx) list"""
        timetable = []
        for module_idx, gene in enumerate(chromosome):
            # gene is int
            if isinstance(gene, tuple):
                gene = gene[0]  # if something slipped through
            room_idx = gene // self.numTimeslots
            timeslot_idx = gene % self.numTimeslots
            timetable.append((module_idx, room_idx, timeslot_idx))
        return timetable

    def __init__(self, hardConstraintPenalty):
        self.hardConstraintPenalty = hardConstraintPenalty

        # Create lecturers
        self.lecturers = [
            Lecturer(0, "Dr. Smith"),
            Lecturer(1, "Prof. Johnson"),
            Lecturer(2, "Dr. Williams"),
            Lecturer(3, "Prof. Brown"),
        ]

        # Create student groups
        self.student_groups = [
            StudentGroup(0, "G1", 40),
            StudentGroup(1, "G2", 35),
            StudentGroup(2, "G3", 40),
            StudentGroup(3, "G4", 20),
        ]

        # Create modules with their assigned lecturers and student groups
        self.courses = [
            Module(0, "Math", self.lecturers[0], self.student_groups[0]),
            Module(1, "Physics", self.lecturers[1], self.student_groups[1]),
            Module(2, "Chemistry", self.lecturers[2], self.student_groups[0]),
            Module(3, "CS", self.lecturers[3], self.student_groups[2]),
            Module(4, "IS", self.lecturers[3], self.student_groups[3]),
            Module(5, "History", self.lecturers[0], self.student_groups[3]),
        ]

        # Create rooms
        self.rooms = [
            Room(0, "R1", 50),
            Room(1, "R2", 30),
            Room(2, "R3", 40),
            Room(3, "R4", 35),
        ]

        # timeslots (4 slots per day: Saturday and Sunday)
        self.timeslots = [
            "Sat-08:30",
            "Sat-10:45",
            "Sat-13:30",
            "Sat-15:45",
            "Sun-08:30",
            "Sun-10:45",
            "Sun-13:30",
            "Sun-15:45",
        ]

        # Define day boundaries (Saturday: slots 0-3, Sunday: slots 4-7)
        self.saturday_slots = [0, 1, 2, 3]
        self.sunday_slots = [4, 5, 6, 7]

        self.numModules = len(self.courses)
        self.numRooms = len(self.rooms)
        self.numTimeslots = len(self.timeslots)

    def __len__(self):
        return self.numModules

    # ========== COST ==========
    def getCost(self, timetable):
        if len(timetable) != self.numModules:
            raise ValueError("timetable size must equal number of modules")

        cost = 0

        formatted_timetable = self.convert_to_timetable_format(timetable)

        # ===== HARD: constraints =====
        module_indices = [m for m, r, t in formatted_timetable]
        if len(set(module_indices)) != self.numModules:
            cost += self.hardConstraintPenalty
        roomClashes = self.countRoomClashes(formatted_timetable)
        lecturerClashes = self.countLecturerClashes(formatted_timetable)
        groupClashes = self.countGroupClashes(formatted_timetable)
        roomCapacityViolations = self.countRoomCapacityViolations(formatted_timetable)
        hardViolations = (
            roomClashes + lecturerClashes + groupClashes + roomCapacityViolations
        )

        # SOFT: constraints
        softViolations = self.countGapsBetweenClasses(formatted_timetable)
        dayOrderViolations = self.countDayOrderViolations(formatted_timetable)

        return (
            self.hardConstraintPenalty * hardViolations
            + softViolations
            + dayOrderViolations
        )

    # ========== HARD ==========
    def countRoomClashes(self, timetable):
        used = {}
        violations = 0
        for m, r, t in timetable:
            if (r, t) in used:
                violations += 1
            else:
                used[(r, t)] = m
        return violations

    def countLecturerClashes(self, timetable):
        used = {}
        violations = 0
        for m, r, t in timetable:
            module = self.courses[m]
            lec_id = module.lecturer.id
            if (lec_id, t) in used:
                violations += 1
            else:
                used[(lec_id, t)] = m
        return violations

    def countGroupClashes(self, timetable):
        used = {}
        violations = 0
        for m, r, t in timetable:
            module = self.courses[m]
            group_id = module.student_group.id
            if (group_id, t) in used:
                violations += 1
            else:
                used[(group_id, t)] = m
        return violations

    def countRoomCapacityViolations(self, timetable):
        violations = 0
        for m, r, t in timetable:
            module = self.courses[m]
            if module.student_group.size > self.rooms[r].capacity:
                violations += 1
        return violations

    # ========== SOFT ==========
    def countGapsBetweenClasses(self, timetable):
        violations = 0
        groupSlots = {}

        for m, r, t in timetable:
            module = self.courses[m]
            group_id = module.student_group.id
            if group_id not in groupSlots:
                groupSlots[group_id] = []
            groupSlots[group_id].append(t)

        for group_id, slots in groupSlots.items():
            slots.sort()
            for i in range(len(slots) - 1):
                if slots[i + 1] - slots[i] > 1:  # gap
                    violations += slots[i + 1] - slots[i] - 1
        return violations

    def countDayOrderViolations(self, timetable):
        """
        Count violations where a student group has classes on Sunday
        but empty slots on Saturday (should fill Saturday first)
        """
        violations = 0

        # Group classes by student group
        group_classes = {}
        for m, r, t in timetable:
            module = self.courses[m]
            group_id = module.student_group.id
            if group_id not in group_classes:
                group_classes[group_id] = []
            group_classes[group_id].append(t)

        # Check each group's schedule
        for group_id, slots in group_classes.items():
            saturday_classes = [t for t in slots if t in self.saturday_slots]
            sunday_classes = [t for t in slots if t in self.sunday_slots]

            # Penalize if group has Sunday classes but empty Saturday slots
            if len(sunday_classes) > 0 and len(saturday_classes) < len(
                self.saturday_slots
            ):
                # More penalty if more Sunday classes and more empty Saturday slots
                violations += len(sunday_classes) * (
                    len(self.saturday_slots) - len(saturday_classes)
                )

        return violations

    # ========== VALIDATION ==========
    def isValidTimetable(self, timetable):
        """Check if the timetable satisfies all hard constraints"""
        if len(timetable) != self.numModules:
            return False
        formatted_timetable = self.convert_to_timetable_format(timetable)

        # Check if all modules are scheduled exactly once
        module_indices = [m for m, r, t in formatted_timetable]
        if len(set(module_indices)) != self.numModules:
            return False

        # Check other hard constraints
        if (
            self.countRoomClashes(formatted_timetable) > 0
            or self.countLecturerClashes(formatted_timetable) > 0
            or self.countGroupClashes(formatted_timetable) > 0
            or self.countRoomCapacityViolations(formatted_timetable) > 0
        ):
            return False

        return True

    # ========== VISUALIZATION ==========
    def printSchedule(self, timetable):
        """
        Prints the schedule and violations details
        :param timetable: a list of (moduleIdx, roomIdx, timeslotIdx)
        """

        print("Schedule for each module:")
        formatted_timetable = self.convert_to_timetable_format(timetable)
        for m, r, t in formatted_timetable:
            module = self.courses[m]
            room = self.rooms[r]
            print(
                f"{module.name} | Lecturer: {module.lecturer.name} | "
                f"Group: {module.student_group.name} (size {module.student_group.size}) | "
                f"Room: {room.name} (capacity {room.capacity}) | Timeslot: {self.timeslots[t]}"
            )
        print()

        # ===== HARD CONSTRAINTS =====
        formatted_timetable = self.convert_to_timetable_format(timetable)
        module_indices = [m for m, r, t in formatted_timetable]
        allModulesScheduled = len(set(module_indices)) == self.numModules
        print("All modules scheduled exactly once =", allModulesScheduled)

        roomClashes = self.countRoomClashes(formatted_timetable)
        lecturerClashes = self.countLecturerClashes(formatted_timetable)
        groupClashes = self.countGroupClashes(formatted_timetable)
        roomCapacityViolations = self.countRoomCapacityViolations(formatted_timetable)

        print("Room clashes =", roomClashes)
        print("Lecturer clashes =", lecturerClashes)
        print("Group clashes =", groupClashes)
        print("Room capacity violations =", roomCapacityViolations)
        print()

        # Soft constraint violations
        softViolations = self.countGapsBetweenClasses(formatted_timetable)
        dayOrderViolations = self.countDayOrderViolations(formatted_timetable)

        print("Gaps between classes =", softViolations)
        print(
            "Day order violations (Sunday before Saturday filled) =", dayOrderViolations
        )
        print()

        # Overall validity
        print("Timetable is valid =", self.isValidTimetable(formatted_timetable))


# testing the class:
def main():
    problem = TimetablingProblem(10)

    np.random.seed(42)
    timetable = [
        (
            i,
            np.random.randint(problem.numRooms),
            np.random.randint(problem.numTimeslots),
        )
        for i in range(problem.numModules)
    ]

    problem.printSchedule(timetable)
    print("\nTotal Cost =", problem.getCost(timetable))


if __name__ == "__main__":
    main()
