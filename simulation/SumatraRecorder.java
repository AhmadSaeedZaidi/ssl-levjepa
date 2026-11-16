import com.fasterxml.jackson.databind.ObjectMapper;
import edu.tigers.sumatra.Sumatra;
import edu.tigers.sumatra.geometry.Geometry;
import edu.tigers.sumatra.ids.BotID;
import edu.tigers.sumatra.ids.ETeamColor;
import edu.tigers.sumatra.math.vector.IVector3;
import edu.tigers.sumatra.math.vector.Vector3;
import edu.tigers.sumatra.model.SumatraModel;
import edu.tigers.sumatra.referee.proto.SslGcRefereeMessage;
import edu.tigers.sumatra.sim.SimState;
import edu.tigers.sumatra.sim.SimulationHelper;
import edu.tigers.sumatra.sim.SumatraSimulator;
import edu.tigers.sumatra.snapshot.SnapObject;
import edu.tigers.sumatra.snapshot.Snapshot;
import java.lang.reflect.Field;
import java.nio.file.*;
import java.io.*;
import java.util.*;

/** Read-only instrumentation of the pinned 2025 native simulator; no physics is reimplemented. */
public class SumatraRecorder {
    static final ObjectMapper JSON = new ObjectMapper();
    static double[] xyz(IVector3 v, double scale) {
        return v == null ? null : new double[]{v.x()*scale, v.y()*scale, v.z()*scale};
    }
    static Snapshot initial(long seed, int scenario) {
        Random r = new Random(seed);
        double bx = (r.nextDouble()-.5)*3_000, by = (r.nextDouble()-.5)*2_500;
        double angle = (r.nextDouble()-.5)*Math.PI*2;
        double speed = new double[]{0, 1.8, 3.0, 0.7, 0, 2.5}[scenario%6];
        var builder = Snapshot.builder().autoContinue(false)
            .command(SslGcRefereeMessage.Referee.Command.FORCE_START)
            .ball(new SnapObject(Vector3.fromXYZ(bx, by, 0), Vector3.fromXYZ(speed*Math.cos(angle), speed*Math.sin(angle), 0)));
        for (ETeamColor color : new ETeamColor[]{ETeamColor.BLUE, ETeamColor.YELLOW}) {
            double side = color == ETeamColor.BLUE ? -1 : 1;
            for (int id=0; id<6; id++) {
                double x = side*(900 + r.nextDouble()*2_600), y = (r.nextDouble()-.5)*4_800;
                double heading = side < 0 ? 0 : Math.PI;
                if (id == 0) { x=bx+side*(scenario%3==0 ? 130 : 550); y=by; }
                if (id == 1 && scenario%3==2) { x=bx+side*380; y=by+450; }
                builder.bot(BotID.createBotId(id, color), new SnapObject(Vector3.fromXYZ(x,y,heading), Vector3.zero()));
            }
        }
        return builder.build();
    }
    static Map<String,Object> record(SimState state, int episode, String match, long seed, int sample, int scenario) {
        Map<String,Object> row = new LinkedHashMap<>();
        row.put("episode_id", episode); row.put("match_id", match); row.put("seed", seed);
        row.put("sample_idx", sample); row.put("scenario", scenario);
        row.put("field_size_m", new double[]{Geometry.getFieldLength()*.001, Geometry.getFieldWidth()*.001});
        row.put("timestamp_ns", state.getSimTime()); row.put("simulator_frame_id", state.getFrameId());
        row.put("action_interval_ns", new long[]{state.getSimTime()-10_000_000, state.getSimTime()});
        var ball = state.getSimulatedBall().getState();
        row.put("ball", Map.of("position_m", xyz(ball.getPos(), .001), "velocity_mps", xyz(ball.getVel(), .001), "acceleration_mps2", xyz(ball.getAcc(), .001)));
        List<Object> robots = new ArrayList<>();
        for (var bot : state.getSimulatedBots().values()) {
            var s = bot.getState(); var a = bot.getAction();
            Map<String,Object> robot = new LinkedHashMap<>(), action = new LinkedHashMap<>();
            robot.put("id", bot.getBotId().getNumber()); robot.put("team", bot.getBotId().getTeamColor().name());
            robot.put("pose_m_rad", new double[]{s.getPose().getPos().x()*.001, s.getPose().getPos().y()*.001, s.getPose().getOrientation()});
            robot.put("velocity_mps_radps", new double[]{s.getVel().x()*.001, s.getVel().y()*.001, s.getVel().z()});
            robot.put("ball_contact", s.isBarrierInterrupted());
            action.put("mode_xy", a.getModeXY().name()); action.put("mode_w", a.getModeW().name());
            // Preserve upstream action units explicitly: target XY and local XY velocity use millimetres.
            action.put("target_position_mm_rad", xyz(a.getTargetPos(), 1));
            action.put("target_velocity_local_mmps_radps", xyz(a.getTargetVelLocal(), 1));
            action.put("kick_speed_mps", a.getKickSpeed()); action.put("chip", a.isChip());
            action.put("disarm", a.isDisarm()); action.put("dribbler_rpm", a.getDribbleRpm());
            robot.put("action_applied_last_10ms", action); robots.add(robot);
        }
        row.put("robots", robots);
        var kick = state.getLastKickEvent();
        row.put("last_kick_timestamp_ns", kick == null ? null : kick.getTimestamp());
        row.put("referee_command", state.getLatestRefereeMessage() == null ? null : state.getLatestRefereeMessage().getCommand().name());
        return row;
    }
    public static void main(String[] args) throws Exception {
        // args: output.jsonl match_index episode_count samples_per_episode base_seed
        Path output = Path.of(args[0]).toAbsolutePath();
        int matchIndex=Integer.parseInt(args[1]), episodes=Integer.parseInt(args[2]), samples=Integer.parseInt(args[3]);
        long baseSeed=Long.parseLong(args[4]);
        Files.createDirectories(output.getParent());
        if (Files.exists(output)) throw new IOException("Refusing to overwrite " + output);
        Sumatra app = new Sumatra();
        app.setHeadless(true); app.setModuli("ssl_record"); app.setAiBlue(true); app.setAiYellow(true);
        app.run();
        SumatraSimulator sim = SumatraModel.getInstance().getModule(SumatraSimulator.class);
        sim.pause(); sim.setManageBotCount(false);
        // No public full-state accessor exists in this release. Read one pinned field only, while paused.
        Field stateField=SumatraSimulator.class.getDeclaredField("simState"); stateField.setAccessible(true);
        String match=String.format("session_%03d",matchIndex);
        try (BufferedWriter writer=Files.newBufferedWriter(output, StandardOpenOption.CREATE_NEW)) {
            for (int e=0; e<episodes; e++) {
                long seed=baseSeed+matchIndex*10_000L+e;
                SimulationHelper.loadSimulation(initial(seed,e));
                long tick=0;
                for (int sample=0; sample<samples; sample++) {
                    long target=1+Math.round(sample*100.0/15);
                    while (tick<target) {
                        long before=((SimState)stateField.get(sim)).getSimTime();
                        sim.step().orElseThrow().get(); tick++;
                        if (((SimState)stateField.get(sim)).getSimTime()-before != 10_000_000) throw new IllegalStateException("Simulation tick mismatch");
                        SimState state=(SimState)stateField.get(sim);
                        var row=record(state,matchIndex*episodes+e,match,seed,tick==target ? sample : -1,e);
                        row.put("tick_idx",tick);
                        writer.write(JSON.writeValueAsString(row));
                        writer.write("\n");
                    }
                }
                writer.flush();
                System.out.println("Recorded " + match + " episode " + e + " (" + samples + " samples)");
            }
        } catch (Throwable failure) { failure.printStackTrace(); System.exit(1); }
        System.exit(0);
    }
}
