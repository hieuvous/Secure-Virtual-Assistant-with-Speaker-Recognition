-- Run this once in Supabase SQL Editor after scripts/migrate_sqlite_to_supabase.py
-- has inserted explicit SQLite IDs. It advances each identity sequence safely.

SELECT setval(pg_get_serial_sequence('public.users', 'id'),
              COALESCE((SELECT MAX(id) FROM public.users), 1),
              (SELECT COUNT(*) > 0 FROM public.users));
SELECT setval(pg_get_serial_sequence('public.course_rooms', 'id'),
              COALESCE((SELECT MAX(id) FROM public.course_rooms), 1),
              (SELECT COUNT(*) > 0 FROM public.course_rooms));
SELECT setval(pg_get_serial_sequence('public.enrollment_samples', 'id'),
              COALESCE((SELECT MAX(id) FROM public.enrollment_samples), 1),
              (SELECT COUNT(*) > 0 FROM public.enrollment_samples));
SELECT setval(pg_get_serial_sequence('public.tasks', 'id'),
              COALESCE((SELECT MAX(id) FROM public.tasks), 1),
              (SELECT COUNT(*) > 0 FROM public.tasks));
SELECT setval(pg_get_serial_sequence('public.schedules', 'id'),
              COALESCE((SELECT MAX(id) FROM public.schedules), 1),
              (SELECT COUNT(*) > 0 FROM public.schedules));
SELECT setval(pg_get_serial_sequence('public.private_notes', 'id'),
              COALESCE((SELECT MAX(id) FROM public.private_notes), 1),
              (SELECT COUNT(*) > 0 FROM public.private_notes));
SELECT setval(pg_get_serial_sequence('public.speaker_profiles', 'id'),
              COALESCE((SELECT MAX(id) FROM public.speaker_profiles), 1),
              (SELECT COUNT(*) > 0 FROM public.speaker_profiles));
SELECT setval(pg_get_serial_sequence('public.audit_logs', 'id'),
              COALESCE((SELECT MAX(id) FROM public.audit_logs), 1),
              (SELECT COUNT(*) > 0 FROM public.audit_logs));
