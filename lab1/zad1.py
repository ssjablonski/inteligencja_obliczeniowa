import datetime
from math import pi, sin

imie = input("Podaj imię:")
rok = int(input("Podaj rok urodzenia:"))
miesiac = int(input("Podaj miesiąc urodzenia:"))
dzien = int(input("Podaj dzień urodzenia:"))
print("Witaj", imie)

today = datetime.date.today()
birthday = datetime.date(rok, miesiac, dzien)
age = today - birthday
days_lived = age.days

fizyczna = sin(2 * pi * days_lived / 23)
emocjonalna = sin(2 * pi * days_lived / 28)
intelektualna = sin(2 * pi * days_lived / 33)


if fizyczna > 0.5:
    print("Jesteś dzisiaj w świetnej formie, gratulacje!", fizyczna)
else:
    print("Dziś lepiej odpocząć", fizyczna)
    fizyczna_nowa = sin(2 * pi * days_lived+1 / 23)
    print("Ale nie przejmuj się jutro twoja fala będzie rowna:", fizyczna_nowa  )


if emocjonalna > 0.5:
    print("Dziś jest dobry dzień na podjęcie ważnych decyzji", emocjonalna)
else:
    print("Dziś lepiej nie podejmować ważnych decyzji", emocjonalna)
    emocjonalna = sin(2 * pi * days_lived / 28)
    print("Ale nie przejmuj się jutro twoja fala będzie rowna:", emocjonalna  )


if intelektualna > 0.5:
    print("Dziś jest dobry dzień na naukę", intelektualna)
else:
    print("Dziś lepiej trochę odpocząc od nauki", intelektualna)
    intelektualna = sin(2 * pi * days_lived / 33)
    print("Ale nie przejmuj się jutro twoja fala będzie rowna:", intelektualna  )

# program zając około 12 minut
    
# import datetime
# from math import pi, sin

# def get_user_input():
#     name = input("Enter your name:")
#     year = int(input("Enter your birth year:"))
#     month = int(input("Enter your birth month:"))
#     day = int(input("Enter your birth day:"))
#     return name, year, month, day

# def calculate_biorhythms(birth_date):
#     today = datetime.date.today()
#     age = today - birth_date
#     days_lived = age.days

#     physical = sin(2 * pi * days_lived / 23)
#     emotional = sin(2 * pi * days_lived / 28)
#     intellectual = sin(2 * pi * days_lived / 33)

#     return physical, emotional, intellectual, days_lived

# def print_biorhythms(name, physical, emotional, intellectual, days_lived):
#     print(f"Hello, {name}")

#     if physical > 0.5:
#         print("You are in great shape today, congratulations!", physical)
#     else:
#         print("It's better to rest today", physical)
#         physical_new = sin(2 * pi * (days_lived+1) / 23)
#         print("But don't worry, your wave will be equal tomorrow:", physical_new)

#     if emotional > 0.5:
#         print("Today is a good day to make important decisions", emotional)
#     else:
#         print("It's better not to make important decisions today", emotional)
#         emotional = sin(2 * pi * days_lived / 28)
#         print("But don't worry, your wave will be equal tomorrow:", emotional)

#     if intellectual > 0.5:
#         print("Today is a good day to learn", intellectual)
#     else:
#         print("It's better to rest from learning today", intellectual)
#         intellectual = sin(2 * pi * days_lived / 33)
#         print("But don't worry, your wave will be equal tomorrow:", intellectual)

# def main():
#     name, year, month, day = get_user_input()
#     birth_date = datetime.date(year, month, day)
#     physical, emotional, intellectual, days_lived = calculate_biorhythms(birth_date)
#     print_biorhythms(name, physical, emotional, intellectual, days_lived)

# main()

# napiszanie promta i otrzymanie poprawnego rozwiazania zajelo mi okolo 5 minut.
    
